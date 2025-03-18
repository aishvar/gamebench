import random
import itertools
from collections import Counter

# Card and Deck Handling
SUITS = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
RANK_TO_VALUE = {rank: i for i, rank in enumerate(RANKS, start=2)}

class Deck:
    def __init__(self):
        self.cards = [(rank, suit) for suit in SUITS for rank in RANKS]
        random.shuffle(self.cards)

    def draw(self, num=1):
        return [self.cards.pop() for _ in range(num)]

# Player Representation
class Player:
    def __init__(self, name, stack):
        self.name = name
        self.stack = stack
        self.hole_cards = []
        self.current_bet = 0
        self.folded = False
        self.all_in = False

    def bet(self, amount):
        bet_amount = min(amount, self.stack)
        self.stack -= bet_amount
        self.current_bet += bet_amount
        if self.stack == 0:
            self.all_in = True
        return bet_amount

    def fold(self):
        self.folded = True

# Hand Evaluation
class HandEvaluator:
    @staticmethod
    def evaluate_five_card_hand(hand):
        values = sorted([RANK_TO_VALUE[rank] for rank, suit in hand], reverse=True)
        suits = [suit for rank, suit in hand]
        value_counts = Counter(values)
        counts = sorted(value_counts.values(), reverse=True)
        unique_vals = sorted(set(values), reverse=True)
        
        is_flush = len(set(suits)) == 1
        
        # Check for straight (account for Ace-low)
        sorted_vals = sorted(set(values))
        is_straight = False
        straight_high = None
        if len(sorted_vals) >= 5:
            for i in range(len(sorted_vals) - 4):
                window = sorted_vals[i:i+5]
                if window == list(range(window[0], window[0] + 5)):
                    is_straight = True
                    straight_high = window[-1]
            # Ace-low straight check
            if set([14, 2, 3, 4, 5]).issubset(set(values)):
                is_straight = True
                straight_high = 5
        
        # Determine hand rank and tie breakers
        if is_flush and is_straight:
            return (8, (straight_high,))  # Straight Flush
        if 4 in counts:
            quad = [val for val, cnt in value_counts.items() if cnt == 4][0]
            kicker = max([val for val in values if val != quad])
            return (7, (quad, kicker))
        if 3 in counts and 2 in counts:
            trip = max([val for val, cnt in value_counts.items() if cnt == 3])
            pair = max([val for val, cnt in value_counts.items() if cnt == 2])
            return (6, (trip, pair))
        if is_flush:
            return (5, tuple(sorted(values, reverse=True)))
        if is_straight:
            return (4, (straight_high,))
        if 3 in counts:
            trip = [val for val, cnt in value_counts.items() if cnt == 3][0]
            kickers = sorted([val for val in values if val != trip], reverse=True)
            return (3, (trip,) + tuple(kickers))
        if counts.count(2) >= 2:
            pairs = sorted([val for val, cnt in value_counts.items() if cnt == 2], reverse=True)
            kicker = max([val for val in values if val not in pairs])
            return (2, tuple(pairs) + (kicker,))
        if 2 in counts:
            pair = [val for val, cnt in value_counts.items() if cnt == 2][0]
            kickers = sorted([val for val in values if val != pair], reverse=True)
            return (1, (pair,) + tuple(kickers))
        return (0, tuple(values))  # High Card

    @staticmethod
    def best_hand_rank(cards):
        if len(cards) <= 5:
            return HandEvaluator.evaluate_five_card_hand(cards)
        best_rank = (-1, ())
        for combo in itertools.combinations(cards, 5):
            rank = HandEvaluator.evaluate_five_card_hand(list(combo))
            if rank > best_rank:
                best_rank = rank
        return best_rank

    @staticmethod
    def hand_description(rank_tuple):
        rank_names = {
            8: "Straight Flush",
            7: "Four of a Kind",
            6: "Full House",
            5: "Flush",
            4: "Straight",
            3: "Three of a Kind",
            2: "Two Pair",
            1: "One Pair",
            0: "High Card"
        }
        rank_value, tiebreakers = rank_tuple
        desc = rank_names.get(rank_value, "Unknown")
        return f"{desc} {tiebreakers}"

    @staticmethod
    def determine_winner(player1, player2, community_cards):
        hand1 = HandEvaluator.best_hand_rank(player1.hole_cards + community_cards)
        hand2 = HandEvaluator.best_hand_rank(player2.hole_cards + community_cards)
        if hand1 > hand2:
            return player1, hand1, hand2
        elif hand2 > hand1:
            return player2, hand1, hand2
        else:
            return None, hand1, hand2  # Tie

# Poker Game Engine
class PokerGame:
    def __init__(self, player1_name="Player1", player2_name="Player2", starting_stack=1000, small_blind=10, big_blind=20):
        self.deck = Deck()
        self.players = [Player(player1_name, starting_stack), Player(player2_name, starting_stack)]
        self.community_cards = []
        self.pot = 0
        self.current_bet = big_blind
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.active_player = 0  # Index of current active player
        self.last_raiser = None
        self.log = []

    def post_blinds(self):
        self.log.append("Posting blinds.")
        bet0 = self.players[0].bet(self.small_blind)  # Small blind
        bet1 = self.players[1].bet(self.big_blind)    # Big blind
        self.pot += bet0 + bet1
        self.log.append(f"{self.players[0].name} posts small blind: {bet0}")
        self.log.append(f"{self.players[1].name} posts big blind: {bet1}")

    def deal(self):
        self.log.append("Dealing hole cards.")
        self.post_blinds()  # Deduct blinds before dealing cards
        for player in self.players:
            player.hole_cards = self.deck.draw(2)
            self.log.append(f"{player.name} receives {player.hole_cards}")

    def betting_round(self, round_name):
        self.log.append(f"Starting betting round: {round_name}")
        for player in self.players:
            player.current_bet = 0
        self.current_bet = 0
        self.last_raiser = None

        for _ in range(2):  # each player gets one action in this simple simulation
            player = self.players[self.active_player]
            if player.folded or player.all_in:
                self.log.append(f"{player.name} cannot act (folded or all-in).")
                self.active_player = 1 - self.active_player
                continue
            if self.current_bet == 0:
                action = "check"
            else:
                call_amount = self.current_bet - player.current_bet
                if player.stack <= call_amount:
                    action = "call"
                else:
                    action = "call"  # Defaulting to call for simulation
            self.take_action(action)
        self.log.append(f"Ending betting round: {round_name}")

    def post_flop(self):
        self.community_cards = self.deck.draw(3)
        self.log.append(f"Flop: {self.community_cards}")

    def post_turn(self):
        card = self.deck.draw(1)[0]
        self.community_cards.append(card)
        self.log.append(f"Turn: {card} -> Community Cards: {self.community_cards}")

    def post_river(self):
        card = self.deck.draw(1)[0]
        self.community_cards.append(card)
        self.log.append(f"River: {card} -> Community Cards: {self.community_cards}")

    def take_action(self, action, amount=0):
        player = self.players[self.active_player]
        if action == "fold":
            player.fold()
            self.log.append(f"{player.name} folds.")
        elif action == "check":
            self.log.append(f"{player.name} checks.")
        elif action == "call":
            call_amount = self.current_bet - player.current_bet
            bet_made = player.bet(call_amount)
            self.pot += bet_made
            self.log.append(f"{player.name} calls {bet_made}.")
        elif action == "raise":
            call_amount = self.current_bet - player.current_bet
            total_raise = call_amount + amount
            bet_made = player.bet(total_raise)
            self.pot += bet_made
            self.current_bet += amount
            self.last_raiser = self.active_player
            self.log.append(f"{player.name} raises by {amount} (total bet: {self.current_bet}).")
        self.active_player = 1 - self.active_player

    def showdown(self):
        self.log.append("Showdown:")
        winner, hand1, hand2 = HandEvaluator.determine_winner(self.players[0], self.players[1], self.community_cards)
        desc1 = HandEvaluator.hand_description(hand1)
        desc2 = HandEvaluator.hand_description(hand2)
        self.log.append(f"{self.players[0].name}'s hand: {self.players[0].hole_cards} -> {desc1}")
        self.log.append(f"{self.players[1].name}'s hand: {self.players[1].hole_cards} -> {desc2}")
        if winner is None:
            result = f"Tie! Pot of {self.pot} is split."
            self.log.append(result)
            print(result)
        else:
            result = f"{winner.name} wins the pot of {self.pot}!"
            self.log.append(result)
            print(result)
        print("\nGame Log:")
        for entry in self.log:
            print(entry)

# Example Usage
if __name__ == "__main__":
    game = PokerGame()
    game.deal()
    game.betting_round("Pre-flop")
    print(f"Player1 Cards: {game.players[0].hole_cards}")
    print(f"Player2 Cards: {game.players[1].hole_cards}")
    game.post_flop()
    game.betting_round("Post-flop")
    game.post_turn()
    game.betting_round("Post-turn")
    game.post_river()
    game.betting_round("Post-river")
    game.showdown()
