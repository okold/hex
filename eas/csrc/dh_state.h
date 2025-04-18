#pragma once

#include <bitset>
#include <cstdint>
#include <boost/multiprecision/cpp_int.hpp>

#include "base_state.h"
#include "utils.h"

inline std::string dh_xvec_str(const uint8_t *x, const char c) {
  std::string lines[] = {
      "                _____",                     //
      "               /     \\",                   //
      "         _____/   2   \\_____",             //
      "        /     \\       /     \\",           //
      "  _____/   1   \\_____/   5   \\_____",     //
      " /     \\       /     \\       /     \\",   //
      "/   0   \\_____/   4   \\_____/   8   \\",  //
      "\\       /     \\       /     \\       /",  //
      " \\_____/   3   \\_____/   7   \\_____/",   //
      "       \\       /     \\       /",          //
      "        \\_____/   6   \\_____/",           //
      "              \\       /",                  //
      "               \\_____/"                    //
  };

  for (int i = 0; i < 9; ++i) {
    if (!x[i]) continue;

    const uint8_t t = x[i] >> 1;
    const uint32_t sub_row = 7 + ((i / 3) - (i % 3)) * 2;
    const uint32_t sub_col = 4 + ((i % 3) + (i / 3)) * 7;
    lines[sub_row][sub_col - 1] = 't';
    lines[sub_row][sub_col] = '=';
    lines[sub_row][sub_col + 1] = '0' + t;

    if (x[i] & 1) {
      lines[sub_row - 2][sub_col - 2] = c;
      lines[sub_row - 2][sub_col - 1] = c;
      lines[sub_row - 2][sub_col] = c;
      lines[sub_row - 2][sub_col + 1] = c;
      lines[sub_row - 2][sub_col + 2] = c;
      lines[sub_row - 1][sub_col - 3] = c;
      lines[sub_row - 1][sub_col + 3] = c;
      lines[sub_row][sub_col - 3] = c;
      lines[sub_row][sub_col + 3] = c;
      lines[sub_row + 1][sub_col - 2] = c;
      lines[sub_row + 1][sub_col - 1] = c;
      lines[sub_row + 1][sub_col] = c;
      lines[sub_row + 1][sub_col + 1] = c;
      lines[sub_row + 1][sub_col + 2] = c;
    }
  }

  std::string repr;
  for (int i = 0; i < 13; ++i) {
    repr += lines[i];
    repr += '\n';
  }

  return repr;
}

template <bool abrupt, uint32_t board_s = 3>
struct DhState : public BaseState<abrupt, board_s> {
  static constexpr uint32_t num_cells = BaseState<abrupt, board_s>::move_count;
  // new openspiel uses these :upsidedown:
  static constexpr uint32_t bits_per_action = num_cells;
  static constexpr uint32_t longest_sequence = num_cells;

  // static constexpr uint32_t bits_per_action = num_cells + 1;
  // static constexpr uint32_t longest_sequence = num_cells * 2 - 1;
  // white is player 2
  // We store the sequence of actions as 9 one-hot vectors of size 9
  // For the board we store 9 one-hot vector of size 9
  enum cell_state {
    kEmpty = 0,
    kWhiteWest = -3,
    kWhiteEast = -2,
    kWhiteWin = -4,  // impossible here
    kWhite = -1,     // White and not edge connected
    kBlackNorth = 3,
    kBlackSouth = 2,
    kBlackWin = 4,  // impossible here
    kBlack = 1,     // Black and not edge connected
  };

  constexpr static int min_cell_state = -4;
  constexpr static int cell_states = 9;

  // static constexpr uint32_t OPENSPIEL_INFOSTATE_SIZE =
  //     num_cells * cell_states + longest_sequence * (1 + bits_per_action);
  static constexpr uint32_t OPENSPIEL_INFOSTATE_SIZE =
      num_cells * cell_states + longest_sequence * bits_per_action;

  uint8_t winner_recursive(uint8_t pos, uint8_t board_size, uint8_t mode, uint8_t visited[], const uint8_t end_state[]) const {
    const auto &x = this->x;
    static const int8_t ADJACENCY_MATRIX_3[9][6] = {
      {1, 3, -1, -1, -1, -1},
      {0, 2, 3, 4, -1, -1},
      {1, 4, 5, -1, -1, -1},
      {0, 1, 4, 6, -1, -1},
      {1, 2, 3, 5, 6, 7},
      {2, 4, 7, 8, -1, -1},
      {3, 4, 7, -1, -1, -1},
      {4, 5, 6, 8, -1, -1},
      {5, 7, -1, -1, -1, -1}
    };

    static const int8_t ADJACENCY_MATRIX_4[16][6] = {
      {1, 4, -1, -1, -1, -1},
      {0, 2, 4, 5, -1, -1},
      {1, 3, 5, 6, -1, -1},
      {2, 6, 7, -1, -1, -1},
      {0, 1, 5, 8, -1, -1},
      {1, 2, 4, 6, 8, 9},
      {2, 3, 5, 7, 9, 10},
      {3, 6, 10, 11, -1, -1},
      {4, 5, 9, 12, -1, -1},
      {5, 6, 8, 10, 12, 13},
      {6, 7, 9, 11, 13, 14},
      {7, 10, 14, 15, -1, -1},
      {8, 9, 13, -1, -1, -1},
      {9, 10, 12, 14, -1, -1},
      {10, 11, 13, 15, -1, -1},
      {11, 14, -1, -1, -1, -1}
    };

    visited[pos] = 1;

    if (end_state[pos]) {
      return 1;
    }
    const int8_t (*adj)[6];
    if (board_size == 3) {
      adj = ADJACENCY_MATRIX_3;
    }
    else if (board_size == 4) {
      adj = ADJACENCY_MATRIX_4;
    }

    for (int i = 0; i < 6; i++) {
      if (adj[pos][i] == -1) {
        break;
      }
      else if (!visited[adj[pos][i]] &&      // position not visited yet
                x[mode][adj[pos][i]] & 1) {  // position has a token
        if (winner_recursive(adj[pos][i], board_size, mode, visited, end_state)) {
          return 1;
        }
      }
    }

    return 0; // no end reached
  }
  
  uint8_t winner() const {
    const auto &x = this->x;
    uint8_t board_size = this->b_s;     // EDIT THIS
    uint8_t n = board_size * board_size;
    

    static const uint8_t END_STATES_B_3[] = {0, 0, 0, 0, 0, 0, 1, 1, 1};
    static const uint8_t END_STATES_R_3[] = {0, 0, 1, 0, 0, 1, 0, 0, 1};
    static const uint8_t END_STATES_B_4[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1};
    static const uint8_t END_STATES_R_4[] = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};

    const uint8_t* end_states_b;
    const uint8_t* end_states_r;
    
    if (board_size == 3) {
      end_states_b = END_STATES_B_3;
      end_states_r = END_STATES_R_3;
    }
    else if (board_size == 4) {
      end_states_b = END_STATES_B_4;
      end_states_r = END_STATES_R_4;
    }

    for (int i = 0; i < board_size; i++) {
      if (x[0][i] & 1) {
        uint8_t visited[n] = {};
        if (winner_recursive(i, board_size, 0, visited, end_states_b)) {
          return 0;
        }
      }
    }

    for (int i = 0; i < board_size; i++) {
      if (x[1][i * board_size] & 1) {
        uint8_t visited[n] = {};
        if (winner_recursive(i * board_size, board_size, 1, visited, end_states_r)) {
          return 1;
        }
      }
    }

    return 0xff;  // No winner
  } 

  bool is_terminal() const { return winner() != 0xff; }

  std::string to_string() const {
    const auto &x = this->x;
    std::string out;

    if (winner() == 0xff) {
      out +=
          "** It is Player " + std::to_string(this->player() + 1) + "'s turn\n";
    } else {
      out +=
          "** GAME OVER -- Player " + std::to_string(winner() + 1) + " wins\n";
    }
    out += "** Player 1's board:\n";
    out += dh_xvec_str(x[0], 'X');
    out += "\n** Player 2's board:\n";
    out += dh_xvec_str(x[1], 'O');
    return out;
  }

  static void compute_openspiel_infostate(uint8_t player, boost::multiprecision::cpp_int info,
                                          std::span<bool> buf) {
    std::fill(buf.begin(), buf.end(), false);
    PerPlayer<std::bitset<num_cells>> b{};
    int n_actions = 0, m_actions = 0;
    uint32_t mask = (1 << (move_size(num_cells)-1)) - 1; //0b1111 for 3x3

    for (boost::multiprecision::cpp_int i = info; i; i >>= move_size(num_cells), ++n_actions) {
      bool success = i & 1 ? true : false;
      
      uint8_t cell = (((i >> 1) & mask) - 1).convert_to<uint8_t>();
      uint8_t p = success ? player : 1 - player;
      b[p][cell] = true;
    }

    for (boost::multiprecision::cpp_int i = info; i; i >>= move_size(num_cells), ++m_actions){
      uint8_t cell = (((i >> 1) & mask) - 1).convert_to<uint8_t>();
      buf[num_cells * cell_states + (n_actions - 1 - m_actions) * bits_per_action + cell] = true;
    }

    for (auto i = 0; i < num_cells; ++i) {
      buf[i * cell_states + kEmpty - min_cell_state] = !(b[0][i] || b[1][i]);
    }

    {
      // west is 0 3 6
      std::bitset<num_cells> west, east;
      // west[0] = b[1][0];
      // west[3] = b[1][3];
      // west[6] = b[1][6];
      // west[1] = b[1][1] && (b[1][0] || b[1][3]);
      // west[4] = b[1][4] && (b[1][3] || b[1][6]);
      // west[7] = b[1][7] && b[1][6];
      // west[2] = b[1][2] && (west[1] || west[4]);
      // west[5] = b[1][5] && (west[4] || west[7]);
      // west[8] = b[1][8] && west[7];

      // east[2] = b[1][2];
      // east[5] = b[1][5];
      // east[8] = b[1][8];
      // east[1] = b[1][1] && b[1][2];
      // east[4] = b[1][4] && (b[1][2] || b[1][5]);
      // east[7] = b[1][7] && (b[1][5] || b[1][8]);
      // east[0] = b[1][0] && east[1];
      // east[3] = b[1][3] && (east[1] || east[4]);
      // east[6] = b[1][6] && (east[4] || east[7]);

      auto white = b[1] & ~(east | west);

      for (auto i = 0; i < num_cells; ++i) {
        buf[i * cell_states + kWhite - min_cell_state] = white[i];
        buf[i * cell_states + kWhiteEast - min_cell_state] = east[i];
        buf[i * cell_states + kWhiteWest - min_cell_state] = west[i];
      }
    }

    {
      std::bitset<num_cells> north, south;
      // north[0] = b[0][0];
      // north[1] = b[0][1];
      // north[2] = b[0][2];
      // north[3] = b[0][3] && (b[0][0] || b[0][1]);
      // north[4] = b[0][4] && (b[0][1] || b[0][2]);
      // north[5] = b[0][5] && b[0][2];
      // north[6] = b[0][6] && (north[3] || north[4]);
      // north[7] = b[0][7] && (north[4] || north[5]);
      // north[8] = b[0][8] && north[5];

      // south[6] = b[0][6];
      // south[7] = b[0][7];
      // south[8] = b[0][8];
      // south[3] = b[0][3] && b[0][6];
      // south[4] = b[0][4] && (b[0][6] || b[0][7]);
      // south[5] = b[0][5] && (b[0][7] || b[0][8]);
      // south[0] = b[0][0] && south[3];
      // south[1] = b[0][1] && (south[3] || south[4]);
      // south[2] = b[0][2] && (south[4] || south[5]);

      auto black = b[0] & ~(north | south);

      for (auto i = 0; i < num_cells; ++i) {
        buf[i * cell_states + kBlack - min_cell_state] = black[i];
        buf[i * cell_states + kBlackNorth - min_cell_state] = north[i];
        buf[i * cell_states + kBlackSouth - min_cell_state] = south[i];
      }
    }

    // #ifdef DEBUG
    for (int i = 0; i < num_cells; ++i) {
      int s = 0;
      for (int j = 0; j < num_cells; ++j) {
        s += buf[i * cell_states + j];
      }
      assert(s == 1);
    }
    // #endif
  }
};

struct CornerDhState : public DhState<false> {
  CornerDhState() : DhState<false>() {
    x[0][0] = (1 << 1) + 1;
    x[1][8] = (1 << 1) + 1;
    t[0] = 1;
    t[1] = 1;
  }

  uint32_t available_actions() const {
    uint32_t ans = DhState<false>::available_actions();
    return ans & (player() == 0 ? 0b011111111 : 0b111111110);
  }
};