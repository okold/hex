#pragma once

#include <array>
#include <boost/unordered/unordered_flat_map.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <cstdint>
#include <span>
#include <valarray>
#include <vector>

#include "averager.h"
#include "utils.h"

struct InfosetMetadata {
  uint32_t legal_actions;
  uint32_t infoset_id;

  bool operator==(const InfosetMetadata &other) const {
    return legal_actions == other.legal_actions &&
           infoset_id == other.infoset_id;
  }
};

// Maps from infoset to legal action mask
using InfosetMap = boost::unordered_flat_map<boost::multiprecision::cpp_int, InfosetMetadata>;

struct Treeplex {
  InfosetMap infosets;
  std::vector<boost::multiprecision::cpp_int> infoset_keys;
  std::vector<uint32_t> legal_actions;
  std::vector<uint32_t> parent_index;
  uint32_t move_count;

  uint32_t num_infosets() const { return infoset_keys.size(); }
  bool is_valid_vector(ConstRealBuf buf, uint32_t move_count=9) const;
  bool is_valid_strategy(ConstRealBuf buf, uint32_t move_count=9) const;
  void set_uniform(RealBuf buf, uint32_t move_count=9) const;
  void bh_to_sf(RealBuf buf, uint32_t move_count=9) const;
  void sf_to_bh(RealBuf buf, uint32_t move_count=9) const;
  Real br(RealBuf grad, RealBuf strat = std::span<Real>(), uint32_t move_count=9) const;
  void regret_to_bh(RealBuf buf, uint32_t move_count=9) const;
};

struct EvExpl {
  Real ev0;
  // gradient of utility wrt the player strategies
  PerPlayer<std::valarray<Real>> gradient;
  // expl[0] is how exploitable player 0 is by a best-responding player 1
  PerPlayer<Real> expl;
  // best_response[0] is the best response to player 1's strategy
  PerPlayer<std::valarray<Real>> best_response;
  uint32_t move_count;
};

template <typename T> struct Traverser {
  PerPlayer<std::shared_ptr<Treeplex>> treeplex;
  PerPlayer<std::valarray<Real>> gradients;
  Traverser();

  void compute_gradients(const PerPlayer<ConstRealBuf> strategies);
  EvExpl ev_and_exploitability(const PerPlayer<ConstRealBuf> strategies);
  Averager new_averager(const uint8_t player, const AveragingStrategy avg);
  void compute_openspiel_infostate(const uint8_t player, size_t i, std::span<bool> buf) const;
  void compute_openspiel_infostates(const uint8_t player, std::span<bool> buf) const;

private:
  PerPlayer<std::array<std::valarray<Real>, (T::move_count)>> bufs_;
  PerPlayer<std::valarray<Real>> sf_strategies_;

  void compute_sf_strategies_(const PerPlayer<ConstRealBuf> strategies);
};
