from numpy.random import default_rng
from dataclasses import dataclass

RANK_SCORES = {}


def basic_strategy(player_hand: list[int], dealer_card: int, is_pair: bool):
    pass


def hand_score(hand: list[int]) -> tuple[int, bool]:
    """Return total score and if it soft or hard score"""
    score = 0
    aces = 0

    for card in hand:
        if card == 0:  # Ace
            score += 11
            aces += 1
        elif card >= 9:  # 10, J, Q, K
            score += 10
        else:
            score += card + 1  # 2 to 9

    # Reduce Aces from 11 → 1 if score > 21
    while score > 21 and aces > 0:
        score -= 10
        aces -= 1

    return score, aces > 0


class BJSimulation:

    @dataclass(frozen=True)
    class Configuration:
        SIMULATION_ROUNDS: int = 1
        N_DECKS: int = 8
        DECK_PEN: float = 0.5
        SHUFFLE_SEED: int = 42

    @dataclass
    class RoundSeatStatistics:
        round_id: int = -1
        seat_index: int = -1
        is_dealer: bool = False
        prev_cards_left: int = 0
        prev_running_count: int = 0
        prev_true_count: float = 0.0
        # Count of the remaining cards by rank
        prev_rank_densities: dict[str, int] = {rank_id: 0 for rank_id in range(0, 13)}
        outcome_win: bool = False

    class Player:
        def __init__(self, seat_index: int):
            self.seat_index = seat_index
            self.hands = []

    def __init__(self, config: Configuration):
        self.config = config

        self.round_stats: list[BJSimulation.RoundSeatStatistics] = []

        # (id of ending round, 0 -> N, the number of rounds until a re-shuffle)
        self.current_round_id: int = 0

        # Initialize RNG used to shuffle cards
        self.rng = default_rng(self.config.SHUFFLE_SEED)

        # Initialize deck of cards and shuffle it
        self.reset_deck()

        # Initialize players at their seat index, including dealer
        # seat 7 is the dealer, seat 0 to 6 are players from right to left.
        self.players = [BJSimulation.Player(seat_index) for seat_index in range(0, 8)]

    def calculate_round_stats(self) -> None: ...

    def create_deck(self) -> list[int]:
        # 0 = A, 1 = 2, ... 12 = K
        return [i for i in range(0, 13) for _ in range(4)] * self.config.N_DECKS

    def shuffle_deck(self) -> None:
        self.rng.shuffle(self.deck)

    def reset_deck(self) -> None:
        # Create and shuffle a new deck
        self.deck = self.create_deck()
        self.shuffle_deck()

    def deck_needs_reset(self) -> bool:
        # Check if we reached deck penetration
        return len(self.deck) < (self.config.N_DECKS * 52) * self.config.DECK_PEN

    def deal_initial_cards(self) -> None:

        # Reset player hands
        for player in self.players:
            player.hands = [
                [],
            ]

        # Deal one card to each player two times in a row
        for _ in range(2):
            for seat_index in range(0, 8):
                card = self.deck.pop(0)
                self.players[seat_index].hands[0].append(card)

    def player_split(self) -> None: ...

    def player_turn(
        self, seat_index: int, hand: list[int] = None, is_split: bool = False
    ) -> bool:

        player = self.players[seat_index]

        if hand is None:
            # Allows for recursive split call
            # Take the first base hand
            hand = player.hands[0]

        # Initial player score
        player_score, is_soft = hand_score(hand)

        # Dealer's visible card
        dealer_card = self.players[-1].hands[0][1]

        action = None
        while player_score <= 21 and action != "S":

            is_pair = False
            if not is_split:
                # If we're not in a split, check if this is a pair
                is_pair = len(hand) == 2 and hand[0] == hand[1]

            # Run basic strategy to choose which action the player performs
            action = basic_strategy(hand, dealer_card, is_pair=is_pair)

            if action == "H":
                # Hit
                hand.append(self.deck.pop(0))

                # Recalculate score
                player_score, is_soft = hand_score(hand)
            elif action == "D":
                # Double down
                hand.append(self.deck.pop(0))  # deal exactly one card
                player_score, is_soft = hand_score(hand)
                return
            elif action == "P" and not is_split:
                # Split hands
                player.hands.remove(hand)
                hand_right = [hand[0], self.deck.pop(0)]
                hand_left = [hand[1], self.deck.pop(0)]
                player.hands.extend([hand_right, hand_left])

                self.player_turn(seat_index, hand=hand_right, is_split=True)
                self.player_turn(seat_index, hand=hand_left, is_split=True)
                return

    def dealer_turn(self) -> None:
        dealer = self.players[-1]
        hand = dealer.hands[0]
        score, is_soft = hand_score(hand)

        while score < 17:
            # Hit on 16
            hand.append(self.deck.pop(0))
            score, is_soft = hand_score(hand)

    def simulate_round(self) -> None:
        if self.deck_needs_reset():
            # Reached penetration, reset the deck
            self.reset_deck()
            self.current_round_id = 0

        # Play the round
        self.deal_initial_cards()

        # Check if dealer wins with blackjack
        if self.players[-1].hands[0][1] == 0 and self.players[-1].hands[0][0] > 9:
            return

        for seat_index in range(0, 7):
            self.player_turn(seat_index)

        self.dealer_turn()

    def start(self) -> None:

        # Simulate N rounds
        for sim_round in range(0, self.config.SIMULATION_ROUNDS):
            print(f"Round {sim_round+1}/{self.config.SIMULATION_ROUNDS}")
            self.simulate_round()
            self.calculate_round_stats()

            self.current_round_id += 1


if __name__ == "__main__":
    BJSimulation().start()
