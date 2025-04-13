#include "traverser.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <omp.h>
#include <boost/multiprecision/cpp_int.hpp>

#include "averager.h"
#include "base_state.h"
#include "dh_state.h"
#include "log.h"
#include "pttt_state.h"
#include "utils.h"

using std::size_t;

namespace {
template <typename T> // T must extend base_state
unsigned long discover_infosets_thread(T root, PerPlayer<InfosetMap> *infosets) {
  T stack[100];
  stack[0] = root;
  size_t stack_len = 1;
  unsigned long count = 0;

  while (stack_len) {
    // Pop from stack
    const T s = stack[--stack_len];
    const uint8_t p = s.player();
    ++count;

    if (s.winner() == 0xff) {
      uint32_t a = s.available_actions();
      const boost::multiprecision::cpp_int info = s.get_infoset();
      const InfosetMetadata md{
          .legal_actions = a,
          .infoset_id = UINT32_MAX,
      };
      assert(!(*infosets)[p].count(info) || (*infosets)[p][info] == md);
      (*infosets)[p][info] = md;

      for (int i = 0; i < T::move_count; ++i, a >>= 1) {
        if (a & 1) {
          T ss = s;
          ss.next(i);
          stack[stack_len++] = ss;
          assert(stack_len < 100);
        }
      }
    }
  }

  return count;
}

template <typename T>
void compute_gradients_thread(
    T root, const PerPlayer<uint32_t> init_parent_seqs,
    const PerPlayer<std::shared_ptr<Treeplex>> treeplex,
    PerPlayer<ConstRealBuf> sf_strategies, PerPlayer<RealBuf> gradients) {
  std::tuple<T,                  // Current state
             PerPlayer<uint32_t> // Parent seqs
             >
      stack[100];
  stack[0] = {root, init_parent_seqs};
  size_t stack_len = 1;

  while (stack_len) {
    const auto it = stack[--stack_len];
    const T &s = std::get<0>(it);
    const uint8_t p = s.player();
    const uint8_t w = s.winner();
    const boost::multiprecision::cpp_int infoset = s.get_infoset();
    const PerPlayer<uint32_t> &seqs = std::get<1>(it);

    if (w == 0xff) {
      const uint32_t info_id = treeplex[p]->infosets.at(infoset).infoset_id;
      PerPlayer<uint32_t> new_seqs = seqs;

      uint32_t a = s.available_actions();
      for (uint32_t i = 0; a; ++i, a >>= 1) {
        if (a & 1) {
          T ss = s;
          ss.next(i);
          new_seqs[p] = T::move_count * info_id + i;
          stack[stack_len++] = {ss, new_seqs};
        }
      }
    } else if (w == 0 || w == 1) {
      const Real sign = -2.0 * w + 1.0; // 1 if w == 0 else -1
      gradients[0][seqs[0]] += sign * sf_strategies[1][seqs[1]];
      gradients[1][seqs[1]] -= sign * sf_strategies[0][seqs[0]];
    }
  }
}
} // namespace

bool Treeplex::is_valid_vector(ConstRealBuf buf, uint32_t move_count) const {
  if (buf.size() != num_infosets() * move_count)
    return false;

  for (const auto &it : infosets) {
    const uint32_t i = it.second.infoset_id;
    const uint32_t a = it.second.legal_actions;
    for (uint32_t j = 0; j < move_count; ++j) {
      if (!(a & (1 << j))) {
        if (buf[i * move_count + j] != 0 || !std::isfinite(buf[i * move_count + j]))
          return false;
      }
    }
  }
  return true;
}

bool Treeplex::is_valid_strategy(ConstRealBuf buf, uint32_t move_count) const {
  if (buf.size() != num_infosets() * move_count)
    return false;

  for (const auto &it : infosets) {
    const uint32_t i = it.second.infoset_id;
    const uint32_t a = it.second.legal_actions;
    Real sum = 0;
    for (uint32_t j = 0; j < move_count; ++j) {
      if (buf[i * move_count + j] < 0 || buf[i * move_count + j] > 1 ||
          !std::isfinite(buf[i * move_count + j]))
        return false;

      if (a & (1 << j)) {
        sum += buf[i * move_count + j];
      } else if (buf[i * move_count + j] != 0.) {
        return false;
      }
    }
    if (std::abs(sum - 1.0) > 1e-6)
      return false;
  }

  return true;
}

void Treeplex::set_uniform(RealBuf buf, uint32_t move_count) const {
  for (const auto &it : infosets) {
    const uint32_t i = it.second.infoset_id;
    const uint32_t a = it.second.legal_actions;
    const uint8_t na = __builtin_popcount(a);

    for (uint32_t j = 0; j < move_count; ++j) {
      // !! is important to convert to 0 or 1
      buf[i * move_count + j] = Real(is_valid(a, j)) / na;
    }
  }

  assert(is_valid_strategy(buf, move_count));
}

void Treeplex::bh_to_sf(RealBuf buf, uint32_t move_count) const {
  CHECK(is_valid_strategy(buf, move_count), "Buffer validation fails");

  for (uint32_t i = 1; i < num_infosets(); ++i) {
    const boost::multiprecision::cpp_int info = infoset_keys[i];
    const uint32_t parent = parent_index[i];
    const uint32_t parent_a = parent_action(info, move_count);
    const Real parent_prob = buf[parent * move_count + parent_a];

    for (uint32_t j = 0; j < move_count; ++j) {
      buf[i * move_count + j] *= parent_prob;
    }
  }

  assert(is_valid_vector(buf, move_count));
}

void Treeplex::sf_to_bh(RealBuf buf, uint32_t move_count) const {
  CHECK(is_valid_vector(buf, move_count), "Buffer validation fails");

  for (const auto &it : infosets) {
    const uint32_t i = it.second.infoset_id;
    const uint32_t a = it.second.legal_actions;
    Real s = 0;
    for (uint32_t j = 0; j < move_count; ++j) {
      s += (buf[i * move_count + j]) * is_valid(a, j);
    }
    if (s < SMALL) {
      s = 0;
      for (uint32_t j = 0; j < move_count; ++j) {
        buf[i * move_count + j] = is_valid(a, j);
        s += buf[i * move_count + j];
      }
    }
    for (uint32_t j = 0; j < move_count; ++j) {
      buf[i * move_count + j] = buf[i * move_count + j] / s;
    }
  }

  assert(is_valid_strategy(buf, move_count));
}

Real Treeplex::br(RealBuf buf, RealBuf strat, uint32_t move_count) const {
  std::fill(strat.begin(), strat.end(), 0.0);

  CHECK(is_valid_vector(buf, move_count), "Buffer validationa fails");
  if (!strat.empty()) {
    CHECK(is_valid_vector(buf, move_count),
          "Buffer validation for destination strategy fails");
  }

  Real max_val = std::numeric_limits<Real>::lowest();
  for (int32_t i = num_infosets() - 1; i >= 0; --i) {
    const boost::multiprecision::cpp_int info = infoset_keys[i];
    const uint32_t mask = legal_actions[i];

    max_val = std::numeric_limits<Real>::lowest();
    uint8_t best_action = 0xff;
    for (uint32_t j = 0; j < move_count; ++j) {
      if ((mask & (1 << j)) && (buf[i * move_count + j] > max_val)) {
        best_action = j;
        max_val = buf[i * move_count + j];
      }
    }
    assert(best_action != 0xff);

    if (i) {
      const uint32_t parent = parent_index[i];
      const uint32_t parent_a = parent_action(info);
      buf[parent * move_count + parent_a] += max_val;
    }
    if (!strat.empty()) {
      strat[i * move_count + best_action] = 1.0;
    }
  }

  if (!strat.empty()) {
    assert(is_valid_strategy(strat, move_count));
  }

  return max_val;
}

void Treeplex::regret_to_bh(RealBuf buf, uint32_t move_count) const {
  CHECK(is_valid_vector(buf, move_count), "Buffer validation fails");

  for (const auto &it : infosets) {
    const uint32_t i = it.second.infoset_id;
    const uint32_t a = it.second.legal_actions;
    relu_normalize(buf.subspan(i * move_count), a, move_count);
  }

  assert(is_valid_strategy(buf, move_count));
}

template <typename T> Traverser<T>::Traverser() {
  for (auto p : {0, 1}) {
    treeplex[p] = std::make_shared<Treeplex>();
    treeplex[p]->infosets.reserve(5000000);
  }

  { // Insert data for root infoset
    T root;
    treeplex[0]->infosets[root.get_infoset()] = InfosetMetadata{
        .legal_actions = root.available_actions(), .infoset_id = UINT32_MAX};
    uint8_t a = 0;
    for (a = 0; a < T::move_count && !is_valid(root.available_actions(), a); ++a)
      ;
    root.next(a);
    assert(root.player() == 1);
    treeplex[1]->infosets[root.get_infoset()] = InfosetMetadata{
        .legal_actions = root.available_actions(), .infoset_id = UINT32_MAX};
  }

  INFO("discovering infosets (num threads: %d)...", omp_get_max_threads());
  unsigned long count = 10;
  const T temp{};
#pragma omp parallel for reduction(+ : count)
  for (int i = 0; i < T::move_count * T::move_count; ++i) {
    T s{};
    {
      const uint8_t a = i % T::move_count;
      assert(treeplex[0]->infosets.count(s.get_infoset()));
      if (!is_valid(s.available_actions(), a))
        continue;
      s.next(a);
    }
    {
      const uint8_t a = i / T::move_count;
      assert(s.player() == 1 && treeplex[1]->infosets.count(s.get_infoset()));
      if (!is_valid(s.available_actions(), a))
        continue;
      s.next(a);
    }

    PerPlayer<InfosetMap> thread_infosets;
    const unsigned long thread_count =
        ::discover_infosets_thread(s, &thread_infosets);
    count += thread_count;

    INFO("  > thread %02d found %.2fM infosets (%.2fB nodes)", i,
         (thread_infosets[0].size() + thread_infosets[1].size()) / 1e6,
         thread_count / 1e9);

#pragma omp critical
    {
      for (auto p : {0, 1})
        treeplex[p]->infosets.insert(thread_infosets[p].begin(),
                                     thread_infosets[p].end());
    }
  }
  INFO("... discovery terminated. Found %.2fM infosets across %.2fB nodes",
       (treeplex[0]->infosets.size() + treeplex[1]->infosets.size()) / 1e6,
       count / 1e9);

#ifdef DEBUG
  PerPlayer<boost::multiprecision::cpp_int> root_infoset_keys = {0, 0};
  root_infoset_keys[0] = T{}.get_infoset();
  {
    T state;
    uint8_t a = 0;
    for (a = 0; a < state.move_count && !is_valid(state.available_actions(), a); ++a)
      ;
    state.next(a);
    root_infoset_keys[1] = state.get_infoset();
  }
  INFO("root infosets: %s, %s", root_infoset_keys[0].str().c_str(), root_infoset_keys[1].str().c_str());
  for (auto p : {0, 1}) {
    INFO("checking infosets of player %d...", p);
    for (const auto &it : treeplex[p]->infosets) {
      const boost::multiprecision infoset = it.first;
      if (infoset != root_infoset_keys[p]) {
        const boost::multiprecision parent = parent_infoset(infoset);
        CHECK(parent <= infoset, "Parent infoset is greater than child");
        CHECK(treeplex[p]->infosets.count(parent),
              "Parent infoset %ld (%s) of %ld (%s) not found", parent,
              infoset_desc(parent).c_str(), infoset,
              infoset_desc(infoset).c_str());
        CHECK(treeplex[p]->infosets[parent].legal_actions &
                  (1u << parent_action(infoset)),
              "Parent action is illegal");
      }
    }
  }
  INFO("... all infosets are consistent");
#endif

#pragma omp parallel for
  for (int p = 0; p < 2; ++p) {
    treeplex[p]->infoset_keys.reserve(treeplex[p]->infosets.size());
    treeplex[p]->parent_index.resize(treeplex[p]->infosets.size());
    treeplex[p]->legal_actions.resize(treeplex[p]->infosets.size());

    for (auto &it : treeplex[p]->infosets) {
      treeplex[p]->infoset_keys.push_back(it.first);
    }
    INFO("sorting infoset and assigning indices for player %d...", p + 1);
    std::sort(treeplex[p]->infoset_keys.begin(),
              treeplex[p]->infoset_keys.end());
    for (uint32_t i = 0; i < treeplex[p]->infoset_keys.size(); ++i) {
      const boost::multiprecision::cpp_int infoset = treeplex[p]->infoset_keys[i];
      assert(treeplex[p]->infosets.count(infoset));

      auto &info = treeplex[p]->infosets[infoset];
      info.infoset_id = i;
      if (i) {
        treeplex[p]->parent_index[i] =
            treeplex[p]->infosets[parent_infoset(infoset)].infoset_id;
      }
      treeplex[p]->legal_actions[i] = info.legal_actions;
    }
  }

  for (auto player : {0, 1}) {
    assert(treeplex[player]->infoset_keys.size() ==
               treeplex[player]->infosets.size() &&
           treeplex[player]->parent_index.size() ==
               treeplex[player]->infosets.size());

    for (int i = 0; i < T::move_count; ++i) {
      bufs_[player][i].resize(treeplex[player]->num_infosets() * T::move_count);
    }

    gradients[player].resize(treeplex[player]->num_infosets() * T::move_count, 0.0);
    sf_strategies_[player].resize(treeplex[player]->num_infosets() * T::move_count, 0.0);
  }

  INFO("... all done.");
}

template <typename T>
void Traverser<T>::compute_gradients(const PerPlayer<ConstRealBuf> strategies) {
  INFO("begin gradient computation (num threads: %d)...",
       omp_get_max_threads());
  for (auto p : {0, 1}) {
    CHECK(treeplex[p]->is_valid_strategy(strategies[p], T::move_count), "Invalid strategy");
    CHECK(treeplex[p]->is_valid_vector(gradients[p], T::move_count),
          "Buffer validation fails");
  }

  compute_sf_strategies_(strategies);
  const T temp{};
  for (auto p : {0, 1}) {
    gradients[p] = 0.0;

    for (int i = 0; i < T::move_count; ++i) {
      bufs_[p][i] = 0.0;
    }
  }

  uint32_t num_finished = 0;
#pragma omp parallel for
  for (unsigned i = 0; i < T::move_count * T::move_count; ++i) {
    PerPlayer<uint32_t> parent_seqs = {0, 0};
    T s{};

    if (!is_valid(s.available_actions(), i % T::move_count, T::move_count))
      continue;
    assert(s.player() == 0);
    parent_seqs[0] =
        treeplex[0]->infosets.at(s.get_infoset()).infoset_id * T::move_count + (i % T::move_count);
    s.next(i % T::move_count); // pl1's move

    if (!is_valid(s.available_actions(), i / T::move_count, T::move_count))
      continue;
    assert(s.player() == 1);
    parent_seqs[1] =
        treeplex[1]->infosets.at(s.get_infoset()).infoset_id * T::move_count + (i / T::move_count);
    s.next(i / T::move_count); // pl2's move

    const PerPlayer<RealBuf> thread_gradients = {bufs_[0][i / T::move_count],
                                                 bufs_[1][i % T::move_count]};
    ::compute_gradients_thread(s, parent_seqs, treeplex,
                               {sf_strategies_[0], sf_strategies_[1]},
                               thread_gradients);

#pragma omp critical
    {
      ++num_finished;
      if (num_finished % 10 == 0) {
        INFO("  > %2d/81 threads returned", num_finished);
      }
    }
  }

  INFO("... aggregating thread buffers...");

#pragma omp parallel for
  for (int p = 0; p < 2; ++p) {
    for (int j = 0; j < T::move_count; ++j) {
      assert(treeplex[p]->is_valid_vector(bufs_[p][j], T::move_count));
      gradients[p] += bufs_[p][j];
    }
  }

  INFO("... all done.");
#ifndef NDEBUG
  for (auto p : {0, 1}) {
    assert(treeplex[p]->is_valid_vector(gradients[p], T::move_count));
  }
#endif
}

template <typename T>
EvExpl
Traverser<T>::ev_and_exploitability(const PerPlayer<ConstRealBuf> strategies) {
  EvExpl out;
  const T temp{};

  INFO("begin exploitability computation...");
  compute_gradients(strategies);
  for (auto p : {0, 1})
    out.gradient[p] = gradients[p];

  INFO("computing expected value...");
  Real ev0 = dot(sf_strategies_[0], gradients[0]);
  Real ev1 = dot(sf_strategies_[1], gradients[1]);
  out.ev0 = (ev0 - ev1) / 2;

  CHECK(std::abs(ev0 + ev1) < 1e-3, "Expected values differ: %.6f != %.6f", ev0,
        ev1);
  out.expl = {ev0, -ev0};
  for (auto p : {0, 1})
    out.best_response[p].resize(treeplex[p]->num_infosets() * T::move_count, 0.0);

  INFO("computing exploitabilities...");
#pragma omp parallel for
  for (int p = 0; p < 2; ++p) {
    out.expl[1 - p] += treeplex[p]->br(gradients[p], out.best_response[p], T::move_count);
  }

  INFO("... all done. (ev0 = %.6f, expl = %.6f, %.6f)", ev0, out.expl[0],
       out.expl[1]);
  return out;
}

template <typename T>
void Traverser<T>::compute_openspiel_infostate(const uint8_t p, size_t i,
                                               std::span<bool> buf) const {
  boost::multiprecision::cpp_int info = treeplex[p]->infoset_keys[i];
  T::compute_openspiel_infostate(p, info, buf);
}

template <typename T>
void Traverser<T>::compute_openspiel_infostates(const uint8_t p,
                                                std::span<bool> buf) const {
  CHECK(p == 0 || p == 1, "player must be 0 or 1 (found %u)", p);
  const uint32_t ncols = T::OPENSPIEL_INFOSTATE_SIZE;
  const uint32_t nrows = treeplex[p]->num_infosets();
  std::fill(buf.begin(), buf.end(), false);

#pragma omp parallel for
  for (uint32_t i = 0; i < nrows; ++i) {
    compute_openspiel_infostate(p, i, buf.subspan(i * ncols, ncols));
  }
}

template <typename T>
Averager Traverser<T>::new_averager(const uint8_t player,
                                    const AveragingStrategy avg) {
  CHECK(player == 0 || player == 1, "Invalid player %d", player);
  return Averager(treeplex[player], avg, T::move_count);
}

template <typename T>
void Traverser<T>::compute_sf_strategies_(
    const PerPlayer<ConstRealBuf> strategies) {
#pragma omp parallel for
  for (int p = 0; p < 2; ++p) {
    std::copy(strategies[p].begin(), strategies[p].end(),
              std::begin(sf_strategies_[p]));
    treeplex[p]->bh_to_sf(sf_strategies_[p], T::move_count);
  }
}

template struct Traverser<DhState<false>>;
template struct Traverser<DhState<true>>;
template struct Traverser<CornerDhState>;
template struct Traverser<PtttState<false>>;
template struct Traverser<PtttState<true>>;
