import streamlit as st
import random
from collections import Counter
from functools import lru_cache
import math

# ------------------------------
# Constants and mappings
# ------------------------------
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
VALUES = {r: int(r) if r.isdigit() else 10 for r in RANKS}
VALUES['A'] = 11  # initial value, will be softened later
RANK_TO_IDX = {r: i for i, r in enumerate(RANKS)}

# ------------------------------
# Helper functions for hand evaluation
# ------------------------------
def hand_total(hand):
    """Return (best total, is_soft) for a list of ranks."""
    total = 0
    aces = 0
    for r in hand:
        if r == 'A':
            aces += 1
            total += 11
        else:
            total += VALUES[r]
    # soften aces if necessary
    while total > 21 and aces > 0:
        total -= 10
        aces -= 1
    soft = (aces > 0)
    return total, soft

def is_soft(hand):
    return hand_total(hand)[1]

# ------------------------------
# Deck management
# ------------------------------
def initial_counts(num_decks):
    """Return list of counts for each rank (13 ranks)."""
    return [4 * num_decks] * 13

def remove_cards(counts, cards):
    """Return new counts after removing given cards (list of ranks)."""
    counts = list(counts)
    for r in cards:
        idx = RANK_TO_IDX[r]
        if counts[idx] <= 0:
            raise ValueError(f"No hay suficientes {r} en el mazo")
        counts[idx] -= 1
    return tuple(counts)

def list_from_counts(counts):
    """Expand counts into a list of ranks (for Monte Carlo)."""
    deck = []
    for i, cnt in enumerate(counts):
        deck.extend([RANKS[i]] * cnt)
    return deck

# ------------------------------
# Exact dealer distribution (recursive with memoization)
# ------------------------------
@lru_cache(maxsize=None)
def dealer_distribution(hand, remaining_counts, soft17_hit):
    """
    Return a dict {17: prob, 18: prob, ..., 'bust': prob}
    hand: tuple of ranks (e.g., ('10', 'A'))
    remaining_counts: tuple of 13 ints
    soft17_hit: bool (True if dealer hits soft 17)
    """
    total, soft = hand_total(hand)
    # stand conditions
    if total > 21:
        return {'bust': 1.0}
    if total >= 17:
        if total == 17 and soft and soft17_hit:
            pass  # continue
        else:
            # stand
            return {total: 1.0} if total <= 21 else {'bust': 1.0}
    
    # otherwise draw a card
    total_cards = sum(remaining_counts)
    if total_cards == 0:
        # no cards left, dealer stands with current total (should not happen normally)
        return {total: 1.0}
    
    result = {}
    for i, cnt in enumerate(remaining_counts):
        if cnt == 0:
            continue
        r = RANKS[i]
        prob = cnt / total_cards
        new_hand = hand + (r,)
        new_counts = list(remaining_counts)
        new_counts[i] -= 1
        new_counts = tuple(new_counts)
        sub_dist = dealer_distribution(new_hand, new_counts, soft17_hit)
        for score, p in sub_dist.items():
            result[score] = result.get(score, 0.0) + prob * p
    return result

def dealer_distribution_from_upcard(upcard, remaining_counts, soft17_hit):
    """Convenience wrapper starting with only upcard."""
    return dealer_distribution((upcard,), remaining_counts, soft17_hit)

# ------------------------------
# Exact EV calculations
# ------------------------------
def ev_stand(player_total, dealer_dist):
    """
    dealer_dist: dict from dealer_distribution (scores or 'bust')
    Returns (ev, win_prob, loss_prob, push_prob)
    """
    if player_total > 21:
        return -1.0, 0.0, 1.0, 0.0
    win = loss = push = 0.0
    for score, prob in dealer_dist.items():
        if score == 'bust':
            win += prob
        else:
            if player_total > score:
                win += prob
            elif player_total < score:
                loss += prob
            else:
                push += prob
    ev = win - loss  # push contributes 0
    return ev, win, loss, push

def ev_hit(player_hand, player_counts, upcard, remaining_counts, soft17_hit):
    """
    Expected value of hitting exactly one card and then standing.
    player_hand: list of ranks (tuple)
    player_counts: counts of ranks in player hand (not needed but kept for consistency)
    remaining_counts: counts after removing player hand and upcard
    """
    total_remaining = sum(remaining_counts)
    if total_remaining == 0:
        return -1.0, 0.0, 1.0, 0.0  # no cards to hit -> effectively stand? but treat as loss

    ev = 0.0
    # aggregate win/loss/push for Kelly later
    agg_win = agg_loss = agg_push = 0.0

    for i, cnt in enumerate(remaining_counts):
        if cnt == 0:
            continue
        r = RANKS[i]
        prob = cnt / total_remaining
        new_hand = player_hand + [r]
        new_total, _ = hand_total(new_hand)
        if new_total > 21:
            # bust
            ev += prob * (-1.0)
            agg_loss += prob
        else:
            # new remaining after hit
            new_rem = list(remaining_counts)
            new_rem[i] -= 1
            new_rem = tuple(new_rem)
            dealer_dist = dealer_distribution_from_upcard(upcard, new_rem, soft17_hit)
            sub_ev, sub_win, sub_loss, sub_push = ev_stand(new_total, dealer_dist)
            ev += prob * sub_ev
            agg_win += prob * sub_win
            agg_loss += prob * sub_loss
            agg_push += prob * sub_push

    return ev, agg_win, agg_loss, agg_push

# ------------------------------
# Monte Carlo simulation
# ------------------------------
def monte_carlo_stand(player_total, upcard, remaining_counts, soft17_hit, n_iter):
    """Simulate dealer only, compare with player_total."""
    win = loss = push = 0
    deck = list_from_counts(remaining_counts)
    for _ in range(n_iter):
        random.shuffle(deck)
        # dealer draws cards
        hand = [upcard]
        # we need to track drawn cards to remove from deck copy
        # simpler: simulate drawing from deck without replacement using index
        # we'll just pop from a copy
        deck_copy = deck.copy()
        # remove upcard from deck_copy (since upcard is already accounted in remaining_counts, but in deck it's included)
        # Actually deck contains all remaining cards including the upcard? No, remaining_counts is after removing upcard, so deck is correct.
        # So we start with empty hand, then add upcard as first card. But we need to simulate dealer draw from deck.
        # Better: start with hand = [upcard], and then draw from deck (which does not contain upcard). That's correct.
        hand = [upcard]
        while True:
            total, soft = hand_total(hand)
            if total > 21:
                # bust
                break
            if total >= 17:
                if total == 17 and soft and soft17_hit:
                    pass
                else:
                    break
            # draw next card
            if not deck_copy:
                break  # no cards left
            next_card = deck_copy.pop()
            hand.append(next_card)
        final_total, _ = hand_total(hand)
        if final_total > 21:
            # dealer bust
            win += 1
        else:
            if player_total > final_total:
                win += 1
            elif player_total < final_total:
                loss += 1
            else:
                push += 1
    ev = (win - loss) / n_iter
    return ev, win/n_iter, loss/n_iter, push/n_iter

def monte_carlo_hit(player_hand, upcard, remaining_counts, soft17_hit, n_iter):
    """Simulate hit then stand."""
    win = loss = push = 0
    deck = list_from_counts(remaining_counts)
    for _ in range(n_iter):
        random.shuffle(deck)
        # player hits one card
        if not deck:
            # no cards to hit, treat as stand
            # but for consistency, just continue with same hand? Actually impossible if remaining_counts>0
            continue
        hit_card = deck.pop()
        new_hand = player_hand + [hit_card]
        player_total, _ = hand_total(new_hand)
        if player_total > 21:
            loss += 1
            continue
        # dealer plays
        dealer_hand = [upcard]
        deck_copy = deck.copy()  # remaining after hit card
        while True:
            total, soft = hand_total(dealer_hand)
            if total > 21:
                break
            if total >= 17:
                if total == 17 and soft and soft17_hit:
                    pass
                else:
                    break
            if not deck_copy:
                break
            dealer_hand.append(deck_copy.pop())
        dealer_total, _ = hand_total(dealer_hand)
        if dealer_total > 21:
            win += 1
        else:
            if player_total > dealer_total:
                win += 1
            elif player_total < dealer_total:
                loss += 1
            else:
                push += 1
    total_iter = n_iter  # if deck empty? we already handled
    ev = (win - loss) / total_iter
    return ev, win/total_iter, loss/total_iter, push/total_iter

# ------------------------------
# Streamlit UI (en español)
# ------------------------------
def main():
    st.set_page_config(page_title="Analizador Exacto de Blackjack", layout="wide")
    st.title("♠️ Análisis Exacto de Blackjack y Monte Carlo ♥️")
    st.markdown("EV matemáticamente preciso y decisión óptima basada en probabilidades exactas.")

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuración")
        num_decks = st.number_input("Número de mazos", min_value=1, max_value=8, value=1, step=1)
        soft17_hit = st.checkbox("El dealer pide con 17 blando", value=False)
        mc_iterations = st.number_input("Iteraciones de Monte Carlo", min_value=1000, max_value=10**6, value=100000, step=1000)

        st.header("💰 Banca")
        if 'bank' not in st.session_state:
            st.session_state.bank = 1000
        if 'bet' not in st.session_state:
            st.session_state.bet = 10
        st.session_state.bank = st.number_input("Banca actual", value=st.session_state.bank, step=100)
        st.session_state.bet = st.number_input("Apuesta actual", value=st.session_state.bet, step=5)

        if st.button("Reiniciar historial"):
            st.session_state.history = []
        if 'history' not in st.session_state:
            st.session_state.history = []

    # Main area: input cards
    st.header("🃏 Estado de la mesa")

    col1, col2 = st.columns(2)
    with col1:
        dealer_upcard = st.selectbox("Carta visible del dealer", RANKS, index=8)  # default 10

    with col2:
        st.markdown("**Cartas del jugador**")
        if 'player_cards' not in st.session_state:
            st.session_state.player_cards = ['10', '7']  # ejemplo

        # Display current cards
        st.write("Mano: " + ", ".join(st.session_state.player_cards))
        total, soft = hand_total(st.session_state.player_cards)
        st.write(f"Total: {total} ({'Blanda' if soft else 'Dura'})")

        # Buttons to add/remove cards
        add_col, rem_col = st.columns(2)
        with add_col:
            new_card = st.selectbox("Añadir carta", RANKS, key="add_card")
            if st.button("➕ Añadir"):
                st.session_state.player_cards.append(new_card)
                st.rerun()
        with rem_col:
            if st.button("➖ Quitar última"):
                if st.session_state.player_cards:
                    st.session_state.player_cards.pop()
                    st.rerun()

    # Validate counts
    total_decks_counts = initial_counts(num_decks)
    visible_cards = st.session_state.player_cards + [dealer_upcard]
    try:
        remaining_counts = remove_cards(total_decks_counts, visible_cards)
    except ValueError as e:
        st.error(f"❌ {e}. Ajusta las cartas o aumenta el número de mazos.")
        st.stop()

    st.success(f"✅ Cartas restantes: {sum(remaining_counts)}")

    if total > 21:
        st.error("¡La mano del jugador ya se pasó! No se puede continuar.")
        st.stop()

    # Execute analysis
    if st.button("🚀 Ejecutar análisis", type="primary"):
        with st.spinner("Calculando probabilidades exactas y Monte Carlo..."):
            # Exact calculations
            dealer_dist = dealer_distribution_from_upcard(dealer_upcard, remaining_counts, soft17_hit)
            ev_stand_val, win_stand, loss_stand, push_stand = ev_stand(total, dealer_dist)
            ev_hit_val, win_hit, loss_hit, push_hit = ev_hit(
                st.session_state.player_cards,
                None,  # not used
                dealer_upcard,
                remaining_counts,
                soft17_hit
            )

            # Monte Carlo
            mc_stand_ev, mc_stand_w, mc_stand_l, mc_stand_push = monte_carlo_stand(
                total, dealer_upcard, remaining_counts, soft17_hit, mc_iterations
            )
            mc_hit_ev, mc_hit_w, mc_hit_l, mc_hit_push = monte_carlo_hit(
                st.session_state.player_cards, dealer_upcard, remaining_counts, soft17_hit, mc_iterations
            )

            # Recommendation based on exact EV
            if ev_stand_val > ev_hit_val:
                recommendation = "PLANTARSE"
                best_ev = ev_stand_val
                best_win, best_loss = win_stand, loss_stand
            else:
                recommendation = "PEDIR"
                best_ev = ev_hit_val
                best_win, best_loss = win_hit, loss_hit

            # Kelly fraction (even money, ignoring pushes)
            kelly_fraction = best_win - best_loss  # p - q

            # Display results
            st.header("📊 Resultados")

            col_res1, col_res2, col_res3 = st.columns(3)
            with col_res1:
                st.metric("Total jugador", f"{total} ({'Blanda' if soft else 'Dura'})")
                st.metric("Prob. de que el dealer se pase", f"{dealer_dist.get('bust', 0):.4%}")

            with col_res2:
                st.metric("EV(Plantarse)", f"{ev_stand_val:.4%}")
                st.metric("EV(Pedir)", f"{ev_hit_val:.4%}")

            with col_res3:
                st.metric("Recomendación", recommendation)
                st.metric("Fracción de Kelly", f"{kelly_fraction:.4%}")

            st.subheader("Distribución exacta del dealer")
            dist_display = {k: f"{v:.4%}" for k, v in dealer_dist.items()}
            st.json(dist_display)

            st.subheader("🔁 Comparación Monte Carlo")
            comp_data = {
                "Método": ["Exacto Plantarse", "MC Plantarse", "Exacto Pedir", "MC Pedir"],
                "EV": [f"{ev_stand_val:.4%}", f"{mc_stand_ev:.4%}", f"{ev_hit_val:.4%}", f"{mc_hit_ev:.4%}"],
                "Ganar": [f"{win_stand:.4%}", f"{mc_stand_w:.4%}", f"{win_hit:.4%}", f"{mc_hit_w:.4%}"],
                "Perder": [f"{loss_stand:.4%}", f"{mc_stand_l:.4%}", f"{loss_hit:.4%}", f"{mc_hit_l:.4%}"],
                "Empate": [f"{push_stand:.4%}", f"{mc_stand_push:.4%}", f"{push_hit:.4%}", f"{mc_hit_push:.4%}"],
            }
            st.dataframe(comp_data, use_container_width=True)

            # Optional: update bank history (simple)
            if st.button("Registrar resultado (Ganar/Perder)"):
                # Not implemented in detail, just a placeholder
                st.session_state.history.append({
                    "mano": st.session_state.player_cards.copy(),
                    "decision": recommendation,
                    "ev": best_ev
                })
                st.success("Registrado.")

if __name__ == "__main__":
    main()
