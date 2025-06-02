import streamlit as st
import functools
import math
import random
import pandas as pd

st.set_page_config(layout="wide")

# Session State for results table
if "results_table" not in st.session_state:
    st.session_state["results_table"] = None

###############################################################################
# 1) TOP UI: Mode, SV, etc.
###############################################################################
colA, colB, colC = st.columns([1,1,1])

with colA:
    # Default load => '30 Prog'
    game_mode = st.selectbox(
        "Side-Game Format",
        ["30 Prog","N+3","N+4","Dino","Infinite"],
        index=0
    )

with colB:
    # If 30Prog => SV=2, else=3
    default_sv = 2.0 if game_mode=="30 Prog" else 3.0
    squid_value = st.number_input("Squid Value", value=default_sv, step=0.5, min_value=0.0)

with colC:
    if game_mode=="30 Prog":
        splashV= st.number_input("SplashV", value=20, step=1, min_value=0)
        bonus_trigger= st.number_input("Bonus Trigger", value=10, step=1, min_value=1)
        bonus_amount= st.number_input("Bonus Amount (bonus_amount*SV=bonus from each player)",
                                      value=10, step=1, min_value=1)
    else:
        splashV=0
        bonus_trigger=0
        bonus_amount=0

###############################################################################
# 2) NUMBER OF PLAYERS
###############################################################################
N = st.number_input("Number of Players (N)", value=8, min_value=1, step=1)

###############################################################################
# Horizontal line
###############################################################################
st.markdown("<hr style='margin:0; height:1px; background-color:#ccc; border:none;' />",
            unsafe_allow_html=True)

###############################################################################
# SEAT LABELS
###############################################################################
def get_seat_labels(num_seats):
    if num_seats<4:
        return [f"Player{i+1}" for i in range(num_seats)]
    neg_count= num_seats-4
    neg_lbls= [f"-{x}" for x in range(neg_count,0,-1)]
    out= neg_lbls + ["BTN","SB","BB","S"]
    return out[:num_seats]

seat_labels= get_seat_labels(N)

###############################################################################
# 3) SEAT UI: tokens + lostcount
###############################################################################
h_current=[]
seat_lost_count=[]

for i in range(N):
    row= st.columns([2,2,1])
    with row[0]:
        st.write(seat_labels[i])
    with row[1]:
        if game_mode=="Dino":
            val= st.radio(f"{seat_labels[i]} tokens",[0,1],
                          horizontal=True, index=0, key=f"dino_{i}")
        else:
            val= st.number_input(f"{seat_labels[i]} tokens",
                                 value=0, step=1, min_value=0,
                                 key=f"dist_{i}")
        h_current.append(val)
    lostkey= f"lostcount_{i}"
    if lostkey not in st.session_state:
        st.session_state[lostkey]= 0
    with row[2]:
        if game_mode=="30 Prog":
            lv= st.selectbox("LC",[0,1,2],
                             index= st.session_state[lostkey],
                             key= lostkey,
                             help="Consecutive lost games (0..2). If 2 => double penalty")
        else:
            lv= st.session_state[lostkey]
        seat_lost_count.append(lv)

###############################################################################
# threshold_map => for N+3 or N+4 if you want Tui or other thresholds
###############################################################################
threshold_map= {}

###############################################################################
# BACKEND LOGIC
###############################################################################
import math
import functools

def seat_value_nplusX(h, thr_map_tuple, sq_val):
    thr_map= dict(thr_map_tuple)
    mm=1.0
    for (th,mul) in sorted(thr_map.items()):
        if h>=th and mul> mm:
            mm= mul
    return h* mm* sq_val

###############################################################################
# PAYOFF singled out & multiple zero => used by N+3, N+4
###############################################################################
def payoff_singled_out_nplusX(h_list, zero_idx, thr_map_tuple, sq_val):
    n_len= len(h_list)
    seatvals= [ seat_value_nplusX(h, thr_map_tuple, sq_val) for h in h_list]
    total_val= sum(seatvals)
    out= [0]* n_len
    out[zero_idx]= -total_val
    for i in range(n_len):
        if i!=zero_idx:
            out[i]= seatvals[i]
    return tuple(out)

def payoff_multiple_zero_nplusX(h_list, thr_map_tuple, sq_val):
    n_len= len(h_list)
    seatvals= [ seat_value_nplusX(h, thr_map_tuple, sq_val) for h in h_list]
    total_val= sum(seatvals)
    zc= sum(x==0 for x in h_list)
    out= [0]* n_len
    for i in range(n_len):
        if h_list[i]==0:
            out[i]= -total_val
        else:
            out[i]= seatvals[i]* zc
    return tuple(out)

###############################################################################
# N+3
###############################################################################
@functools.lru_cache(None)
def compute_ev_Nplus3(h_tuple, T_left, p_tuple, n_len, thr_map_tuple, sq_val, first_pot):
    h_list= list(h_tuple)
    zc= sum(x==0 for x in h_list)
    if T_left<=0:
        if zc==0:
            return (0,)*n_len
        elif zc==1:
            zidx= [i for i,x in enumerate(h_list) if x==0][0]
            return payoff_singled_out_nplusX(h_list,zidx, thr_map_tuple, sq_val)
        else:
            return payoff_multiple_zero_nplusX(h_list, thr_map_tuple, sq_val)
    if zc==1:
        zidx= [i for i,x in enumerate(h_list) if x==0][0]
        return payoff_singled_out_nplusX(h_list,zidx, thr_map_tuple, sq_val)

    out= [0]* n_len
    distribution= p_tuple if first_pot else tuple([1.0/n_len]*n_len)
    for w in range(n_len):
        pw= distribution[w]
        if pw>0:
            newh= h_list[:]
            newh[w]+=1
            sub= compute_ev_Nplus3(tuple(newh), T_left-1, p_tuple, n_len,
                                   thr_map_tuple, sq_val, False)
            for i in range(n_len):
                out[i]+= pw* sub[i]
    return tuple(out)

###############################################################################
# N+4
###############################################################################
@functools.lru_cache(None)
def compute_ev_Nplus4(h_tuple, T_left, p_tuple, n_len, thr_map_tuple, sq_val, first_pot):
    h_list= list(h_tuple)
    zc= sum(x==0 for x in h_list)
    if T_left<=0:
        if zc==0:
            return (0,)*n_len
        elif zc==1:
            zidx= [i for i,x in enumerate(h_list) if x==0][0]
            return payoff_singled_out_nplusX(h_list,zidx, thr_map_tuple, sq_val)
        else:
            return payoff_multiple_zero_nplusX(h_list, thr_map_tuple, sq_val)
    if zc==1:
        zidx= [i for i,x in enumerate(h_list) if x==0][0]
        return payoff_singled_out_nplusX(h_list,zidx, thr_map_tuple, sq_val)

    out= [0]* n_len
    distribution= p_tuple if first_pot else tuple([1.0/n_len]*n_len)
    for w in range(n_len):
        pw= distribution[w]
        if pw>0:
            newh= h_list[:]
            newh[w]+=1
            sub= compute_ev_Nplus4(tuple(newh), T_left-1, p_tuple, n_len,
                                   thr_map_tuple, sq_val, False)
            for i in range(n_len):
                out[i]+= pw* sub[i]
    return tuple(out)

###############################################################################
# Dino
###############################################################################
@functools.lru_cache(None)
def compute_ev_Dino(h_tuple, p_tuple, n_len, sq_val, first_pot):
    h_list= list(h_tuple)
    zc= sum(x==0 for x in h_list)
    if zc==1:
        zidx= [i for i,x in enumerate(h_list) if x==0][0]
        # singled-out
        holders= sum(x==1 for x in h_list)
        total_val= holders* sq_val
        out= [0]* n_len
        out[zidx]= -total_val
        for i in range(n_len):
            if i!=zidx and h_list[i]==1:
                out[i]= sq_val
        return tuple(out)
    if zc==0:
        return (0,)*n_len

    distribution= p_tuple if first_pot else tuple([1.0/n_len]*n_len)
    probH=0.0
    probZ=0.0
    zero_idx=[]
    for i in range(n_len):
        if h_list[i]==1:
            probH+= distribution[i]
        else:
            probZ+= distribution[i]
            zero_idx.append(i)
    if probH>=1.0 or probZ<=0:
        return (0,)*n_len

    sumEV= [0]* n_len
    for zpos in zero_idx:
        relw= distribution[zpos]/ probZ
        newh= h_list[:]
        newh[zpos]=1
        sub= compute_ev_Dino(tuple(newh), p_tuple, n_len, sq_val, False)
        for k in range(n_len):
            sumEV[k]+= relw* sub[k]
    factor= probZ/(1.0- probH)
    out= [factor*x for x in sumEV]
    return tuple(out)

###############################################################################
# Infinite
###############################################################################
@functools.lru_cache(None)
def compute_ev_Infinite(h_tuple, p_tuple, n_len, sq_val, first_pot):
    h_list= list(h_tuple)
    zc= sum(x==0 for x in h_list)
    if zc==1:
        zidx= [i for i,x in enumerate(h_list) if x==0][0]
        total_t= sum(h_list)
        total_val= total_t* sq_val
        out= [0]* n_len
        out[zidx]= -total_val
        for i in range(n_len):
            if i!=zidx:
                out[i]= h_list[i]* sq_val
        return tuple(out)
    if zc==0:
        return (0,)*n_len

    distribution= p_tuple if first_pot else tuple([1.0/n_len]*n_len)
    probH=0.0
    probZ=0.0
    zero_idx=[]
    for i in range(n_len):
        if h_list[i]>0:
            probH+= distribution[i]
        else:
            probZ+= distribution[i]
            zero_idx.append(i)
    if probH>=1.0 or probZ<=0:
        return (0,)*n_len

    sumEV= [0]* n_len
    for zpos in zero_idx:
        relw= distribution[zpos]/ probZ
        newh= h_list[:]
        newh[zpos]+=1
        sub= compute_ev_Infinite(tuple(newh), p_tuple, n_len, sq_val, False)
        for k in range(n_len):
            sumEV[k]+= relw* sub[k]
    factor= probZ/(1.0- probH)
    out= [factor*x for x in sumEV]
    return tuple(out)

###############################################################################
# 30 Prog => MONTE CARLO ONLY
###############################################################################
def scenario_ev_30Prog_MC(h_list, seat_i_forced, forced_win,
                          sq_val, spV, b_trig, b_amt, lost_arr):
    """
    We'll forcibly set seat_i_forced to win or lose pot #1. Then random awarding for rest.
    Return final payoff lumpsum array for 1 run.
    """
    n_len= len(h_list)
    used_tokens= sum(h_list)
    # forced pot #1
    if forced_win:
        h_list[seat_i_forced]+=1
    else:
        # seat_i_forced loses => pot #1 goes randomly among other seats
        others= []
        for s in range(n_len):
            if s!= seat_i_forced:
                others.append(s)
        chosen= random.choice(others)
        h_list[chosen]+=1
    # now random awarding until 30 used or exactly 1 seat <2
    while True:
        sub2= [i for i,x in enumerate(h_list) if x<2]
        if len(sub2)==1:
            break
        used_tokens= sum(h_list)
        if used_tokens>=30:
            break
        r= random.randrange(n_len)
        h_list[r]+=1

    # final => compute payoffs
    return payoff_30Prog_final(h_list, lost_arr, spV, b_trig, b_amt, sq_val)

def payoff_30Prog_final(h_list, lost_arr, spV, b_trig, b_amt, sq_val):
    n_len= len(h_list)
    out= [0]* n_len
    losers= [i for i,x in enumerate(h_list) if x<2]
    bonus_seats= [i for i,x in enumerate(h_list) if x>= b_trig]

    # Step A: bonus
    for i in range(n_len):
        if i not in bonus_seats:
            cost=0
            for j in bonus_seats:
                if j!= i:
                    cost+= b_amt* sq_val
                    out[j]+= b_amt* sq_val
            out[i]-= cost

    # Step B: splash
    for L_i in losers:
        portion= spV/ float(n_len)
        for seat_j in range(n_len):
            out[seat_j]+= portion
        out[L_i]-= spV

    # Step C: (winner_squids-1)*sv, double if lost_arr[L_i]>=2
    winners= [w for w,x in enumerate(h_list) if x>=2]
    for L_i in losers:
        dbl=2 if lost_arr[L_i]>=2 else 1
        for w_j in winners:
            if w_j!=L_i:
                cost= (h_list[w_j]-1)* sq_val* dbl
                out[L_i]-= cost
                out[w_j]+= cost
    return out

def compute_ev_30Prog_MC(h_tuple, seat_i_forced, forced_win,
                         sq_val, spV, b_trig, b_amt, lost_arr,
                         R=20000):
    """
    Do R random runs => average lumpsum payoff
    """
    base_list= list(h_tuple)
    n_len= len(base_list)
    sumPay= [0]* n_len
    for _ in range(R):
        copyh= base_list[:]  # new copy
        finalP= scenario_ev_30Prog_MC(copyh, seat_i_forced, forced_win,
                                      sq_val, spV, b_trig, b_amt, lost_arr)
        for i in range(n_len):
            sumPay[i]+= finalP[i]
    avg= [x/R for x in sumPay]
    return tuple(avg)

###############################################################################
# scenario_ev_R => router for other modes
###############################################################################
@functools.lru_cache(None)
def scenario_ev_R(h_tuple, mode, p_tuple, thr_map_tuple, sq_val,
                  first_pot, spV, b_trig, b_amt, lost_arr):
    h_list= list(h_tuple)
    n_len= len(h_list)
    if mode=="N+3":
        T_left= (n_len+3)- sum(h_list)
        return compute_ev_Nplus3(tuple(h_list), T_left, p_tuple, n_len,
                                 thr_map_tuple, sq_val, first_pot)
    elif mode=="N+4":
        T_left= (n_len+4)- sum(h_list)
        return compute_ev_Nplus4(tuple(h_list), T_left, p_tuple, n_len,
                                 thr_map_tuple, sq_val, first_pot)
    elif mode=="Dino":
        return compute_ev_Dino(tuple(h_list), p_tuple, n_len, sq_val, first_pot)
    elif mode=="Infinite":
        return compute_ev_Infinite(tuple(h_list), p_tuple, n_len, sq_val, first_pot)
    else:
        # 30Prog => we won't do recursion here. We'll do MC in scenario_ev_A/B
        return (0,)* n_len

###############################################################################
# scenario_ev_A => forcibly seat_i wins => normal recursion for old modes,
# or MonteCarlo for 30Prog
###############################################################################
@functools.lru_cache(None)
def scenario_ev_A(h_tuple, mode, p_tuple, seat_i, thr_map_tuple, sq_val,
                  first_pot, spV, b_trig, b_amt, lost_arr):
    h_list= list(h_tuple)
    n_len= len(h_list)
    if mode=="30 Prog":
        # do MonteCarlo
        return compute_ev_30Prog_MC(tuple(h_list), seat_i, True,
                                    sq_val, spV, b_trig, b_amt, lost_arr,
                                    R=20000)
    elif mode=="N+3":
        T_left= (n_len+3)- sum(h_list)
        if T_left<=0:
            return scenario_ev_R(tuple(h_list), mode, p_tuple, thr_map_tuple,
                                 sq_val, False, spV,b_trig,b_amt,lost_arr)
        h_list[seat_i]+=1
        return compute_ev_Nplus3(tuple(h_list), T_left-1, p_tuple, n_len,
                                 thr_map_tuple, sq_val, False)
    elif mode=="N+4":
        T_left= (n_len+4)- sum(h_list)
        if T_left<=0:
            return scenario_ev_R(tuple(h_list), mode, p_tuple, thr_map_tuple,
                                 sq_val, False, spV,b_trig,b_amt,lost_arr)
        h_list[seat_i]+=1
        return compute_ev_Nplus4(tuple(h_list), T_left-1, p_tuple, n_len,
                                 thr_map_tuple, sq_val, False)
    elif mode=="Dino":
        if h_list[seat_i]==0:
            h_list[seat_i]=1
        return compute_ev_Dino(tuple(h_list), p_tuple, n_len, sq_val, False)
    elif mode=="Infinite":
        h_list[seat_i]+=1
        return compute_ev_Infinite(tuple(h_list), p_tuple, n_len, sq_val, False)
    return (0,)* n_len

###############################################################################
# scenario_ev_B => forcibly seat_i loses => normal recursion or MC for 30Prog
###############################################################################
@functools.lru_cache(None)
def scenario_ev_B(h_tuple, mode, p_tuple, seat_i, thr_map_tuple, sq_val,
                  first_pot, spV, b_trig, b_amt, lost_arr):
    h_list= list(h_tuple)
    n_len= len(h_list)
    if mode=="30 Prog":
        # do MonteCarlo
        return compute_ev_30Prog_MC(tuple(h_list), seat_i, False,
                                    sq_val, spV, b_trig, b_amt, lost_arr,
                                    R=20000)
    if mode=="N+3":
        T_left= (n_len+3)- sum(h_list)
        if T_left<=0:
            return scenario_ev_R(tuple(h_list), mode, p_tuple, thr_map_tuple,
                                 sq_val, False, spV,b_trig,b_amt,lost_arr)
        accum=[0]* n_len
        sum_others= 1.0- p_tuple[seat_i]
        if sum_others<=0:
            return scenario_ev_R(tuple(h_list), mode, p_tuple, thr_map_tuple,
                                 sq_val, False, spV,b_trig,b_amt,lost_arr)
        for other in range(n_len):
            if other== seat_i: continue
            wprob= p_tuple[other]/ sum_others
            alt_h= h_list[:]
            alt_h[other]+=1
            sub= compute_ev_Nplus3(tuple(alt_h), T_left-1, p_tuple, n_len,
                                   thr_map_tuple, sq_val, False)
            for k in range(n_len):
                accum[k]+= wprob* sub[k]
        return tuple(accum)
    elif mode=="N+4":
        T_left= (n_len+4)- sum(h_list)
        if T_left<=0:
            return scenario_ev_R(tuple(h_list), mode, p_tuple, thr_map_tuple,
                                 sq_val, False, spV,b_trig,b_amt,lost_arr)
        accum=[0]* n_len
        sum_others= 1.0- p_tuple[seat_i]
        if sum_others<=0:
            return scenario_ev_R(tuple(h_list), mode, p_tuple, thr_map_tuple,
                                 sq_val, False, spV,b_trig,b_amt,lost_arr)
        for other in range(n_len):
            if other== seat_i: continue
            wprob= p_tuple[other]/ sum_others
            alt_h= h_list[:]
            alt_h[other]+=1
            sub= compute_ev_Nplus4(tuple(alt_h), T_left-1, p_tuple, n_len,
                                   thr_map_tuple, sq_val, False)
            for k in range(n_len):
                accum[k]+= wprob* sub[k]
        return tuple(accum)
    elif mode=="Dino":
        accum=[0]* n_len
        sum_others= 1.0- p_tuple[seat_i]
        if sum_others<=0:
            return (0,)* n_len
        for other in range(n_len):
            if other== seat_i: continue
            wprob= p_tuple[other]/ sum_others
            alt_h= h_list[:]
            if alt_h[other]==0:
                alt_h[other]=1
            sub= compute_ev_Dino(tuple(alt_h), p_tuple, n_len, sq_val, False)
            for k in range(n_len):
                accum[k]+= wprob* sub[k]
        return tuple(accum)
    elif mode=="Infinite":
        accum=[0]* n_len
        sum_others= 1.0- p_tuple[seat_i]
        if sum_others<=0:
            return (0,)* n_len
        for other in range(n_len):
            if other== seat_i: continue
            wprob= p_tuple[other]/ sum_others
            alt_h= h_list[:]
            alt_h[other]+=1
            sub= compute_ev_Infinite(tuple(alt_h), p_tuple, n_len, sq_val, False)
            for k in range(n_len):
                accum[k]+= wprob* sub[k]
        return tuple(accum)
    return (0,)* n_len

###############################################################################
# 4) do_compute_incentives => final table
###############################################################################
def do_compute_incentives():
    p_tuple= tuple([1.0/N]* N)
    h_tuple= tuple(h_current)
    thr_tup= tuple(sorted(threshold_map.items()))
    lost_arr= tuple(seat_lost_count)

    results=[]
    for seat_i in range(N):
        evA= scenario_ev_A(h_tuple, game_mode, p_tuple, seat_i, thr_tup, squid_value,
                           True, splashV, bonus_trigger, bonus_amount, lost_arr)
        evB= scenario_ev_B(h_tuple, game_mode, p_tuple, seat_i, thr_tup, squid_value,
                           True, splashV, bonus_trigger, bonus_amount, lost_arr)
        myA= evA[seat_i]
        myB= evB[seat_i]
        seatDiff= myA- myB
        inc_2dp= round(seatDiff,2)
        results.append({
            "Seat": seat_labels[seat_i],
            "EV(A)": round(myA,2),
            "EV(B)": round(myB,2),
            "Incentive (A-B)": inc_2dp,
            "Tokens": h_current[seat_i],
        })
    return results

compute_btn= st.button("Compute Incentives Now")
if compute_btn:
    new_table= do_compute_incentives()
    if new_table is not None:
        st.session_state["results_table"]= new_table

if st.session_state["results_table"] is not None:
    df= pd.DataFrame(st.session_state["results_table"])
    df.set_index("Seat", inplace=True)
    st.dataframe(df, use_container_width=True)
else:
    st.info("Results will appear here after 'Compute Incentives Now'.")
