from dataclasses import dataclass, field
from datetime import datetime
from numpy.random import default_rng, randint
from pandas import read_csv, DataFrame

BASIC_STRATEGY_CSV_PATH = "BasicStrategy.csv"

BASIC_STRATEGY_DATAFRAME = read_csv(BASIC_STRATEGY_CSV_PATH, sep=";", header=None)


def basic_strategy(player_hand: list[int], dealer_card: int, is_pair: bool) -> str:
    """Returns an action (H=Hit, S=Stand, P=Split, Dh/Ds=Double Down)

    Card values need to have an int map as so -> (A=0, 1=2, 2=3, ..., 10=J, 11=Q, 12=K)
    """

    if dealer_card == 0:
        dealer_index = 9  # Pair of Aces action cell y
    elif dealer_card >= 9:
        dealer_index = 8  # Pair of high cards action cell y
    else:
        dealer_index = dealer_card - 1

    if is_pair:
        if player_hand[0] == 0:
            player_index = 36  # Pair of Aces action cell x
        elif player_hand[0] >= 9:
            player_index = 35  # Pair of high cards action cell x
        else:
            player_index = player_hand[0] + 26  # Pair region offset
    else:
        score, is_soft = hand_score(player_hand)
        if is_soft:
            player_index = score + 5  # Soft score region offset
        else:
            player_index = score - 4  # Hard score region offset

    return BASIC_STRATEGY_DATAFRAME.iloc[player_index, dealer_index]


def hand_score(hand: list[int]) -> tuple[int, bool]:
    """Return total score and if it soft or hard score

    Card values need to have an int map as so -> (A=0, 1=2, 2=3, ..., 10=J, 11=Q, 12=K)
    """

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
    """Blackjack Basic Strategy simulation"""

    @dataclass(frozen=True)
    class Configuration:
        SIMULATION_ROUNDS: int = 10
        N_DECKS: int = 8
        DECK_PEN: float = 0.5
        SHUFFLE_SEED: int = 42
        USE_RANDOM_SEED: bool = False

    @dataclass
    class RoundSeatStatistics:
        round_id: int = -1
        seat_index: int = -1
        is_dealer: int = 0
        prev_cards_drawed: int = 0
        prev_cards_left: int = 0
        prev_deck_penetration: float = 0.0
        prev_hilo_count: tuple[int, float] = (0, 0.0)
        # Count of the remaining cards by rank
        prev_rank_densities: dict[str, int] = field(default_factory=dict)
        outcome_win: int = 0

    class Player:
        def __init__(self, seat_index: int):
            self.seat_index = seat_index
            self.hands = []

    def __init__(self, config: Configuration = Configuration()):
        self.config = config

        self.round_stats: list[BJSimulation.RoundSeatStatistics] = []

        # (id of ending round, 0 -> N, the number of rounds until a re-shuffle)
        self.current_round_id: int = 0

        # Initialize RNG used to shuffle cards
        if self.config.USE_RANDOM_SEED:
            self.rng = default_rng(randint(0, 2**16 - 1))
        else:
            self.rng = default_rng(self.config.SHUFFLE_SEED)

        # Initialize game in the initial state
        self.reset_game()

    def calculate_round_stats(self) -> None:
        """Calculate and register this round stats"""

        dealer_score, is_soft = hand_score(self.players[-1].hands[0])

        # Player seats stats
        for seat_index in range(0, 7):
            stats = BJSimulation.RoundSeatStatistics(
                self.current_round_id,
                seat_index,  # seat index
                0,  # It's not the dealer
                self.prev_cards_drawed,  # Cards drawed before round
                self.prev_cards_left,  # Cards left before round
                self.prev_deck_penetration,  # Deck penetration before round
                self.prev_hilo_count,  # Hi Log count before round
                self.prev_rank_densities,  # Rank densities before round
            )

            # Calculate winning outcome for the seat
            player_hands = self.players[seat_index].hands
            if len(player_hands) > 1:
                # Split is won with at least a win+push
                hand_scores = [hand_score(hand)[0] for hand in player_hands]
                stats.outcome_win = int(
                    all(
                        [
                            self.player_wins(score, dealer_score)
                            or self.player_pushes(score, dealer_score)
                            for score in hand_scores
                        ]
                    )
                )
            else:
                # Player wins if it beats the dealer's hand
                # Push = Loss
                stats.outcome_win = int(
                    self.player_wins(hand_score(player_hands[0])[0], dealer_score)
                )

            self.round_stats.append(stats)

        # Dealer's seat stats
        self.round_stats.append(
            BJSimulation.RoundSeatStatistics(
                self.current_round_id,
                7,  # seat index
                1,  # It's the dealer
                self.prev_cards_drawed,  # Cards drawed before round
                self.prev_cards_left,  # Cards left before round
                self.prev_deck_penetration,  # Deck penetration before round
                self.prev_hilo_count,  # Hi Log count before round
                self.prev_rank_densities,  # Rank densities before round
                int(dealer_score <= 21),  # Dealers wins if it doesn't bust
            )
        )

        # Update previous-round stats
        self.prev_cards_drawed = (self.config.N_DECKS * 52) - len(self.deck)
        self.prev_cards_left = len(self.deck)
        self.prev_deck_penetration = round(
            1 - self.prev_cards_left / (self.config.N_DECKS * 52), 2
        )

        self.prev_hilo_count = (self.current_hilo_count[0], self.current_hilo_count[1])
        self.prev_rank_densities = {
            k: v for k, v in self.current_rank_densities.items()
        }

    def update_hi_lo_score(self, card: int) -> None:
        """
        Updates the Hi-Lo running and true count based on the card log.
        """
        running = self.current_hilo_count[0]  # Take current running count

        # Hi-Lo count system
        if card == 0 or (9 <= card <= 12):
            running -= 1  # High card
        elif 1 <= card <= 5:
            running += 1  # Low card

        # Recalculate true count
        decks_remaining = max(len(self.deck) / 52, 1 / 52)  # Avoid division by zero
        true_count = running / decks_remaining

        self.current_hilo_count = (running, round(true_count, 2))

    def update_rank_densities(self, card: int) -> None:
        """Updates the count of remaining cards for each rank"""
        self.current_rank_densities[card] -= 1

    def player_wins(self, player_score: int, dealer_score: int) -> int:
        return player_score <= 21 and (dealer_score > 21 or player_score > dealer_score)

    def player_pushes(self, player_score: int, dealer_score: int) -> int:
        return player_score <= 21 and dealer_score == player_score

    def draw_card(self) -> int:
        """Draw a card from the deck"""
        card = self.deck.pop(0)

        # Update current stats
        self.update_hi_lo_score(card)
        self.update_rank_densities(card)

        return card

    def create_deck(self) -> list[int]:
        """Creates a new deck with this card mapping -> 0 = A, 1 = 2, ... 12 = K"""
        return [i for i in range(0, 13) for _ in range(4)] * self.config.N_DECKS

    def shuffle_deck(self) -> None:
        if self.config.USE_RANDOM_SEED:
            self.rng = default_rng(randint(0, 2**16 - 1))
        self.rng.shuffle(self.deck)

    def reset_game(self) -> None:
        """Create and shuffle a new deck"""
        self.deck = self.create_deck()
        self.shuffle_deck()

        # Initialize players at their seat index, including dealer
        # seat 7 is the dealer, seat 0 to 6 are players from right to left.
        self.players = [BJSimulation.Player(seat_index) for seat_index in range(0, 8)]

        self.prev_cards_drawed = 0
        self.prev_cards_left = len(self.deck)
        self.prev_deck_penetration = 0.0
        self.prev_rank_densities = {
            card_id: 4 * self.config.N_DECKS for card_id in range(0, 13)
        }
        self.prev_hilo_count = (0, 0.0)  # Running count, True Count

        self.current_rank_densities = {
            card_id: 4 * self.config.N_DECKS for card_id in range(0, 13)
        }
        self.current_hilo_count = (0, 0.0)  # Running count, True Count

    def deck_needs_reset(self) -> bool:
        """Check if we reached deck penetration"""
        return len(self.deck) < (self.config.N_DECKS * 52) * self.config.DECK_PEN

    def deal_initial_cards(self) -> None:
        """Reset player hands and deal the initial shoe"""

        for player in self.players:
            player.hands = [
                [],
            ]

        # Deal one card to each player two times in a row
        for _ in range(2):
            for seat_index in range(0, 8):
                card = self.draw_card()
                self.players[seat_index].hands[0].append(card)

    def player_turn(
        self, seat_index: int, hand: list[int] = None, is_split: bool = False
    ) -> None:
        """Simulate player turn"""

        player = self.players[seat_index]

        if hand is None:
            # Allows for recursive split call
            # Take the first base hand
            hand = player.hands[0]

        # Initial player score
        player_score, is_soft = hand_score(hand)

        # Dealer's visible card
        dealer_card = self.players[-1].hands[0][0]

        action = None
        while player_score < 21 and action != "S":

            # If we're not in a split, check if this is a pair
            is_pair = not is_split and len(hand) == 2 and hand[0] == hand[1]

            # Run basic strategy to choose which action the player performs
            action = basic_strategy(hand, dealer_card, is_pair=is_pair)

            if action == "H":
                # Hit
                hand.append(self.draw_card())

                # Recalculate score
                player_score, is_soft = hand_score(hand)
            elif action in {"Dh", "Ds"}:
                # Double down
                hand.append(self.draw_card())  # deal exactly one card
                player_score, is_soft = hand_score(hand)
                return
            elif action == "P" and not is_split:
                # Split hands
                player.hands.remove(hand)
                hand_right = [hand[0], self.draw_card()]
                hand_left = [hand[1], self.draw_card()]
                player.hands.extend([hand_right, hand_left])

                self.player_turn(seat_index, hand=hand_right, is_split=True)
                self.player_turn(seat_index, hand=hand_left, is_split=True)
                return

    def dealer_turn(self) -> None:
        """Simulate dealer turn"""
        dealer = self.players[-1]
        hand = dealer.hands[0]
        score, is_soft = hand_score(hand)

        while score < 17:
            # Hit on 16
            hand.append(self.draw_card())
            score, is_soft = hand_score(hand)

    def simulate_round(self) -> None:
        """Simulate a blackjack round"""
        if self.deck_needs_reset():
            # Reached penetration, reset the deck
            self.reset_game()
            self.current_round_id = 0

        # Play the round
        self.deal_initial_cards()

        # Check if dealer wins with blackjack
        if self.players[-1].hands[0][0] == 0 and self.players[-1].hands[0][1] > 9:
            # Don't consider insurance
            return

        for seat_index in range(0, 7):
            self.player_turn(seat_index)

        self.dealer_turn()

    def start(self) -> None:
        """Run the simulation on the total amount of rounds configured,
        calculating stats for each round"""

        # Simulate N rounds
        for sim_round in range(0, self.config.SIMULATION_ROUNDS):
            print(f"Round {sim_round+1}/{self.config.SIMULATION_ROUNDS}")
            self.simulate_round()
            self.calculate_round_stats()

            self.current_round_id += 1


    def round_stats_to_csv(self, output_csv:str = None) None:

        # Independent variables
        headers = [
            "RoundID",
            "SeatIndex",
            "IsDealer",
            "PrevCardsLeft",
            "PrevCardsDrawed",
            "PrevDeckPen",
            "PrevRunningCount",
            "PrevTrueCount",
        ]

        # Add rank headers using card mapping
        headers.extend(
            [
                {
                    0: "PrevDensityA",
                    1: "PrevDensity2",
                    2: "PrevDensity3",
                    3: "PrevDensity4",
                    4: "PrevDensity5",
                    5: "PrevDensity6",
                    6: "PrevDensity7",
                    7: "PrevDensity8",
                    8: "PrevDensity9",
                    9: "PrevDensity10",
                    10: "PrevDensityJ",
                    11: "PrevDensityQ",
                    12: "PrevDensityK",
                }[card_id]
                for card_id in range(0, 13)
            ]
        )

        # Dependent variables
        headers.append("Outcome")

        # Create dataframe
        data = []
        for stat in self.round_stats:
            # Independent variables
            row = [
                stat.round_id,
                stat.seat_index,
                stat.is_dealer,
                stat.prev_cards_left,
                stat.prev_cards_drawed,
                stat.prev_deck_penetration,
                stat.prev_hilo_count[0],
                stat.prev_hilo_count[1],
            ]
            # Include rank densities from A, 2 ... to K
            row.extend([stat.prev_rank_densities[i] for i in range(0, 13)])

            # Dependent variables
            row.append(stat.outcome_win)
            data.append(row)

        df = DataFrame(data, columns=headers)
        
        if output_csv is None:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_csv = f"bjsim_{timestamp_str}.csv"
        df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    sim = BJSimulation()
    sim.start()
    sim.round_stats_to_csv()
