#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>

#include <vector>

#include <boost/multiprecision/cpp_int.hpp>

const uint8_t TIE = 0xee;

inline uint32_t move_size(int move_count) {
  return 32 - __builtin_clz(move_count) + 1;
  // one bit for the player, others left board location
}

template <bool abrupt, uint32_t board_s = 3> struct BaseState {
  

  uint8_t x[2][board_s*board_s]; // state: 2 players, 9 possible moves each
  // x[p][i] is
  //   0 if player p never played cell i
  //   otherwise x[p][i] >> 1 indicates the turn on which player p played on
  //   cell i and x[p][i] & 1 is 1 if player p was first to play cell i, or 0 if
  //   the opponent had played cell i before
  uint8_t p;    // player (0 or 1)
  uint8_t t[2]; // turn

  static constexpr uint32_t b_s = board_s;
  uint32_t move_count = b_s*b_s;

  BaseState()
      : x{}, p{0},
        t{0, 0} {}

  uint8_t player() const { return p; }

  void next(const uint8_t i) {
    // player p moves on cell i
    ++t[p];              // increment player turn
    x[p][i] = t[p] << 1; // store player's move (leave 1st bit free for below)
    if (x[p ^ 1][i] ==
        0) {        // if the opponent hadn't already played on that cell
      x[p][i] |= 1; // set 1st bit to 1 to indicate that
      if constexpr (!abrupt)
        p ^= 1; // move to next player (unless abrupt then it's handled below)
    } // otherwise don't move to next player (unless abrupt) because we can play
      // again
    if constexpr (abrupt) {
      // In the abrupt variant, the turn
      // always passes to the opponent.
      p ^= 1;
    }
  }

  uint32_t available_actions() const {
    uint32_t actions = 0;
    for (uint32_t i = 0; i < move_count; ++i) {
      if (!x[p][i]) {
        actions |= (1 << i);
      }
    }
    return actions;
  }

  // get the infoset for the *current player*
  // infoset is made up of 64 bits, containing all of the player's t_ moves
  // each move is encoded over 5 bits as follows:
  // - the first bit (`move & 1`) is 1 if the move was successful, or 0 if the
  // opponent had played there before
  // - the next 4 bits contain the move, from 1 to 9 (ie. `((move >> 1) &
  // 0b1111) - 1` contains the cell between 0 and 8) the latest move is encoded
  // in the rightmost 5 bits in total the 5*t_ rightmost bits are used, the
  // others are left to 0
  boost::multiprecision::cpp_int get_infoset() const {
    boost::multiprecision::cpp_int info = 0;
    uint8_t t_ = t[p];
    // int state_req = move_size(move_count);
    for (uint32_t i = 0; i < move_count; ++i) {
      const uint8_t to = x[p][i];
      const uint8_t td = t_ - (x[p][i] >> 1);
      assert(td <= move_count);
      info |= boost::multiprecision::cpp_int(((i + 1) << 1) + (to & 1)) << (move_size(move_count) * td);
    }
    // if a cell hasn't been played, td = t_ and this is stored in the infoset
    // -> there have only been t_ moves played, so set all bits past the
    // (5*t_)'s bit to 0
    info &= (boost::multiprecision::cpp_int(1) << (move_size(move_count) * t_)) - 1;
    return info;
  }

  // void compute_openspiel_infostate(bool *buf) const {
  //   uint64_t info = get_infoset();
  //   const uint32_t nfeatures = 27 + 81;
  //   memset(buf, 0, nfeatures * sizeof(bool));

  //   // Mark first 9 cells as empty
  //   memset(buf, 1, 9 * sizeof(bool));
  //   for (uint32_t j = 0; info; info >>= 5, ++j) {
  //     const uint8_t cell = ((info >> 1) & 0b1111) - 1;
  //     const bool placed = info & 1;

  //     buf[cell] = false;
  //     buf[9 + cell] = ((p + placed) % 2 == 0);  // p1 moves
  //     buf[18 + cell] = ((p + placed) % 2 == 1); // p0 moves
  //     // we are reading moves from latest to oldest, and we want to store moves
  //     // from oldest to latest
  //     buf[27 + 9 * (t[p] - 1 - j) + cell] = true;
  //   }
  // }
};

// get total number of moves played so far
inline uint8_t num_actions(boost::multiprecision::cpp_int infoset, uint64_t move_count=9) {
  uint8_t actions = 0;
  for (; infoset; ++actions, infoset >>= move_size(move_count))
    ;
  return actions;
}

// get the infoset one action before (last action = rightmost bits)
inline boost::multiprecision::cpp_int parent_infoset(boost::multiprecision::cpp_int infoset, uint64_t move_count=9) {
  assert(infoset);
  return infoset >> move_size(move_count);
}

// get the last move (between 0 and 8) that was played
inline uint32_t parent_action(boost::multiprecision::cpp_int infoset, uint64_t move_count=9) {
  assert(infoset);
  uint32_t mask = (1 << (move_size(move_count)-1))-1;
  uint32_t pos = (((infoset >> 1) & mask) - 1).convert_to<uint32_t>();
  return pos;
}

inline std::vector<uint8_t> infoset_xvec(boost::multiprecision::cpp_int infoset, uint64_t move_count=9) {
  const uint8_t na = num_actions(infoset, move_count);
  std::vector<uint8_t> x(move_count);
  for (int i = na; infoset; --i, infoset >>= move_size(move_count)) {
    uint32_t mask = (1 << move_size(move_count))-1;
    const uint8_t co = (infoset & mask).convert_to<uint8_t>();
    assert(co < 2*move_count && i >= 1);
    x[(co >> 1) - 1] = (i << 1) + (co & 1);
  }
  return x;
}

inline std::string infoset_desc(boost::multiprecision::cpp_int key, uint64_t move_count=9) {
  std::string out = "";
  uint32_t mask = (1 << move_size(move_count))-2;
  for (; key; key >>= move_size(move_count)) {
    out += (key & 1) ? '*' : '.';
    out += std::to_string((((key & mask) >> 1) - 1).convert_to<uint32_t>());
  }
  std::reverse(out.begin(), out.end());
  return out;
}