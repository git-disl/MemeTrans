import json
import pandas as pd
from collections import defaultdict
import pytz
import numpy as np
import copy
import matplotlib.pyplot as plt
import random
import pickle as pkl
import os
from tqdm import tqdm
from collections import OrderedDict
import pandas as pd
bundled_meme_cnt = 0
no_mint_cnt = 0

class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

        root = x
        while self.parent[root] != root:
            root = self.parent[root]

        while x != root:
            px = self.parent[x]
            self.parent[x] = root
            x = px
        return root

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return

        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1

def gini(array: np.ndarray) -> float:
    """计算基尼系数（数组必须非负，长度≥1）。"""
    if len(array) == 0:
        return 0.0
    array = np.sort(array.clip(min=0))
    n = len(array)
    cum = np.cumsum(array, dtype=float)
    return (n + 1 - 2 * np.sum(cum) / cum[-1]) / n if cum[-1] > 0 else 0.0

def feature_generation(mint_address, tx_list, price_dict, sig2bundle, account2signer):

    def nz(v):
        return 0 if pd.isna(v) else v

    # if mint_address != "Xf8GE7QjMW41AjpAeoQUqPFs2LEnSArB3tadKy9pump":
    # if mint_address != "66xnURQWTuxgET8EZorJMRQDJCZHVprJZU3y9YHBVC5F":
    #     return None
    if "mint" not in tx_list[0]["type"]: # it is unormal
        global no_mint_cnt
        no_mint_cnt += 1
        return None

    valid_tx_list = []
    for tx in tx_list:
        if "swap" in tx["type"]:
            valid_tx_list.append(tx)

    if len(valid_tx_list) == 0:
        return None

    feat = OrderedDict()
    mint_ts = tx_list[0]["timestamp"]
    ts = tx_list[-1]["timestamp"]
    ts_hour = ts - (ts % 3600)
    migrate_ts = pd.to_datetime(ts, unit="s", utc=True)

    feat["mint_ts"] = mint_ts
    feat["mint_address"] = mint_address

    #========feature group 1: context=========#
    prefix = "group1"
    feat[f"{prefix}_price"] = price_dict[ts_hour]
    feat[f"{prefix}_migrate_year"] = migrate_ts.year
    feat[f"{prefix}_migrate_month"] = migrate_ts.month  # 1–12
    feat[f"{prefix}_migrate_day"] = migrate_ts.day  # 1–31
    feat[f"{prefix}_migrate_hour"] = migrate_ts.hour  # 0–23
    feat[f"{prefix}_migrate_weekday"] = migrate_ts.weekday()  # 0 = Monday

    holdings = defaultdict(int)
    cost = defaultdict(float)
    realized_pnl = defaultdict(float)

    sniper_0s_list = set()
    sniper_1s_list = set()
    sniper_5s_list = set()
    sniper_10s_list = set()
    wash_tx_cnt = 0
    swap_tx_cnt = 0
    transfer_tx_cnt = 0
    dev_initial_amt = 0
    trader_map_tuples = []
    dev_addr_set = set()
    cluster_set_list = []
    bundle2cluster = defaultdict(set)
    signer2cluster = defaultdict(set)

    for row in tx_list:
        if row["type"] == "zero":
            wash_tx_cnt += 1
            continue
        if "swap" in row["type"]:
            swap_tx_cnt += 1
        if "transfer" in row["type"]:
            transfer_tx_cnt += 1

        price = row["price"]

        trader_map = row["trader_map"]
        timestamp = row["timestamp"]

        if len(trader_map)>1 and "swap" in row["type"]: # transfer or swap
            cluster_set_list.append(set(trader_map.keys()))

        if row["signature"] in sig2bundle.keys() and trader_map:
            bundle_id = sig2bundle[row["signature"]]
            bundle2cluster[bundle_id] = bundle2cluster[bundle_id].union(set(trader_map.keys()))

        for trader, value in trader_map.items():
            trader_map_tuples.append([row["type"], trader, value])
            if trader in account2signer.keys():
                signer2cluster[account2signer[trader]].add(trader)

            if value > 0 and "swap" in row["type"]:
                cost[trader] += value * abs(price)
            elif value < 0 and "swap" in row["type"]:
                sell_amount = -value
                if holdings[trader] > 0:
                    avg_cost = cost[trader] / holdings[trader]
                    realized_pnl[trader] += (abs(price) - avg_cost) * sell_amount
                    cost[trader] -= avg_cost * sell_amount
            holdings[trader] += value

            if timestamp == mint_ts and "swap" in row["type"]:
                sniper_0s_list.add(trader)
            if timestamp <= mint_ts + 1 and "swap" in row["type"]:
                sniper_1s_list.add(trader)
            if timestamp <= mint_ts + 5 and "swap" in row["type"]:
                sniper_5s_list.add(trader)
            if timestamp <= mint_ts + 10 and "swap" in row["type"]:
                sniper_10s_list.add(trader)
            if "mint&swap" in row["type"]:
                dev_addr_set.add(trader)
                dev_initial_amt += value

    # calculate unrealized_pnl
    current_price = valid_tx_list[-1]["price"]
    unrealized_pnl = {
        trader: holdings[trader] * (current_price - (cost[trader] / holdings[trader] if holdings[trader] > 0 else 0))
        for trader in holdings
    }

    realized_pnl_df = pd.DataFrame(list(realized_pnl.items()), columns=["trader", "amount"])
    unrealized_pnl_df = pd.DataFrame(list(unrealized_pnl.items()), columns=["trader", "amount"])

    if tx_list[0]["type"] == "mint":
        if tx_list[1]["type"] == "swap":
            for trader, value in tx_list[1]["trader_map"].items():
                dev_addr_set.add(trader)
                dev_initial_amt += value

    if len(holdings) == 0:
        return None

    # merge bundles into cluster info
    for k,v in bundle2cluster.items():
        if len(v)>1:
            cluster_set_list.append(v)

    for k, v in signer2cluster.items():
        if len(v)>1:
            cluster_set_list.append(v)


    hold_df = (pd.DataFrame(holdings.items(), columns=["trader", "amount"])
               .sort_values("amount", ascending=False))
    trader_df = pd.DataFrame(trader_map_tuples, columns=["type", "trader", "value"])
    total_supply = hold_df.amount.sum() or 1e-6

    # # holder stats
    swap_df = trader_df[trader_df['type'].str.contains('swap', case=False, na=False)]
    buy_df = swap_df[swap_df['value'] > 0].copy()
    sell_df = swap_df[swap_df['value'] < 0].copy()
    first_buyers = buy_df.drop_duplicates("trader", keep="first")
    early_top1_buyers = first_buyers.head(1).trader.tolist()
    early_top5_buyers = first_buyers.head(5).trader.tolist()
    early_top10_buyers = first_buyers.head(10).trader.tolist()
    early_top20_buyers = first_buyers.head(20).trader.tolist()
    sniper_0s_holdings = hold_df[hold_df.trader.isin(sniper_0s_list)]
    sniper_1s_holdings = hold_df[hold_df.trader.isin(sniper_1s_list)]
    sniper_5s_holdings = hold_df[hold_df.trader.isin(sniper_5s_list)]
    sniper_10s_holdings = hold_df[hold_df.trader.isin(sniper_10s_list)]
    early_top1_holdings = hold_df[hold_df.trader.isin(early_top1_buyers)]
    early_top5_holdings = hold_df[hold_df.trader.isin(early_top5_buyers)]
    early_top10_holdings = hold_df[hold_df.trader.isin(early_top10_buyers)]
    early_top20_holdings = hold_df[hold_df.trader.isin(early_top20_buyers)]

    # ==========Feature Group 2: holding concentration========
    prefix = "group2"
    feat[f"{prefix}_holder_gini"] = gini(hold_df[hold_df.amount >= 1].amount.values)
    feat[f"{prefix}_top1_pct"] = hold_df.head(1).amount.sum() / total_supply
    feat[f"{prefix}_top5_pct"] = hold_df.head(5).amount.sum() / total_supply
    feat[f"{prefix}_top10_pct"] = hold_df.head(10).amount.sum() / total_supply
    feat[f"{prefix}_top20_pct"] = hold_df.head(20).amount.sum() / total_supply
    feat[f"{prefix}_top50_pct"] = hold_df.head(50).amount.sum() / total_supply
    feat[f"{prefix}_top100_pct"] = hold_df.head(100).amount.sum() / total_supply
    feat[f"{prefix}_top10_to_top100"] = feat[f"{prefix}_top10_pct"] / (feat[f"{prefix}_top100_pct"] + 1e-6)
    feat[f"{prefix}_sniper_0s_hold_pct"] = sniper_0s_holdings.amount.sum() / total_supply
    feat[f"{prefix}_sniper_1s_hold_pct"] = sniper_1s_holdings.amount.sum() / total_supply
    feat[f"{prefix}_sniper_5s_hold_pct"] = sniper_5s_holdings.amount.sum() / total_supply
    feat[f"{prefix}_sniper_10s_hold_pct"] = sniper_10s_holdings.amount.sum() / total_supply
    feat[f"{prefix}_early_top1_hold_pct"] = early_top1_holdings.amount.sum() / total_supply
    feat[f"{prefix}_early_top1_initial_hold_pct"] = first_buyers.head(1).value.sum() / total_supply
    feat[f"{prefix}_early_top1_hold_ratio"] = feat[f"{prefix}_early_top1_hold_pct"] / feat[f"{prefix}_early_top1_initial_hold_pct"]
    feat[f"{prefix}_early_top5_hold_pct"] = early_top5_holdings.amount.sum() / total_supply
    feat[f"{prefix}_early_top5_initial_hold_pct"] = first_buyers.head(5).value.sum() / total_supply
    feat[f"{prefix}_early_top5_hold_ratio"] = feat[f"{prefix}_early_top5_hold_pct"] / feat[f"{prefix}_early_top5_initial_hold_pct"]
    feat[f"{prefix}_early_top10_hold_pct"] = early_top10_holdings.amount.sum() / total_supply
    feat[f"{prefix}_early_top10_initial_hold_pct"] = first_buyers.head(10).value.sum() / total_supply
    feat[f"{prefix}_early_top10_hold_ratio"] = feat[f"{prefix}_early_top10_hold_pct"] / feat[f"{prefix}_early_top10_initial_hold_pct"]
    feat[f"{prefix}_early_top20_hold_pct"] = early_top20_holdings.amount.sum() / total_supply
    feat[f"{prefix}_early_top20_initial_hold_pct"] = first_buyers.head(20).value.sum() / total_supply
    feat[f"{prefix}_early_top20_hold_ratio"] = feat[f"{prefix}_early_top20_hold_pct"] / feat[f"{prefix}_early_top20_initial_hold_pct"]
    feat[f"{prefix}_dev_hold_pct"] = sum(holdings.get(addr, 0) for addr in dev_addr_set) / total_supply
    feat[f"{prefix}_dev_initial_hold_pct"] = dev_initial_amt / total_supply
    feat[f"{prefix}_dev_hold_ratio"] = nz(feat[f"{prefix}_dev_hold_pct"] / dev_initial_amt)
    feat[f"{prefix}_sniper_0s_num"] = len(sniper_0s_list)
    feat[f"{prefix}_sniper_0s_ratio"] = len(sniper_0s_list) / len(hold_df)
    feat[f"{prefix}_sniper_1s_num"] = len(sniper_1s_list)
    feat[f"{prefix}_sniper_1s_ratio"] = len(sniper_1s_list) / len(hold_df)
    feat[f"{prefix}_sniper_5s_num"] = len(sniper_5s_list)
    feat[f"{prefix}_sniper_5s_ratio"] = len(sniper_5s_list) / len(hold_df)
    feat[f"{prefix}_sniper_10s_num"] = len(sniper_10s_list)
    feat[f"{prefix}_sniper_10s_ratio"] = len(sniper_10s_list) / len(hold_df)
    def mean_pnl(df, addr_set):
        if len(addr_set) == 0:
            return 0.0
        sub = df[df["trader"].isin(addr_set)]
        if len(sub) == 0:
            return 0.0
        return sub.iloc[:, 1].mean()

    groups = {
        "early_top5": early_top5_buyers,
        "early_top10": early_top10_buyers,
        "early_top20": early_top20_buyers,
        "sniper_0s": sniper_0s_list,
        "sniper_1s": sniper_1s_list,
        "sniper_5s": sniper_5s_list,
        "sniper_10s": sniper_10s_list,
        "top5": set(hold_df.head(5).trader),
        "top10": set(hold_df.head(10).trader),
        "top20": set(hold_df.head(20).trader),
        "top100": set(hold_df.head(100).trader),
        "all": set(hold_df.trader)
    }
    for name, addr_set in groups.items():
        feat[f"{prefix}_{name}_realized_pnl_mean"] = mean_pnl(realized_pnl_df, addr_set)
        feat[f"{prefix}_{name}_unrealized_pnl_mean"] = mean_pnl(unrealized_pnl_df, addr_set)

    # ===========Feature Group 3: Market Activity============
    prefix = "group3"
    feat[f"{prefix}_tx_num"] = len(tx_list)
    feat[f"{prefix}_tx_num_valid"] = len(valid_tx_list)
    feat[f"{prefix}_trader_tup_num"] = len(trader_map_tuples)
    feat[f"{prefix}_time_span"] = tx_list[-1]["timestamp"] - mint_ts
    feat[f"{prefix}_time_span_valid"] = valid_tx_list[-1]["timestamp"] - valid_tx_list[0]["timestamp"]
    feat[f"{prefix}_tx_per_sec"] = feat[f"{prefix}_tx_num"] / (feat[f"{prefix}_time_span"] + 1e-6)
    feat[f"{prefix}_tx_per_sec_valid"] = feat[f"{prefix}_tx_num_valid"] / (feat[f"{prefix}_time_span_valid"] + 1e-6)
    feat[f"{prefix}_wash_tx_num"] = wash_tx_cnt
    feat[f"{prefix}_wash_ratio"] = wash_tx_cnt / len(tx_list)
    feat[f"{prefix}_transfer_tx_num"] = wash_tx_cnt / len(tx_list)
    feat[f"{prefix}_transfer_ratio"] = transfer_tx_cnt / len(tx_list)
    feat[f"{prefix}_trader_num"] = len(hold_df)
    feat[f"{prefix}_holder_num"] = (hold_df.amount >=1).sum()
    feat[f"{prefix}_buy_num"] = len(buy_df)
    feat[f"{prefix}_sell_num"] = len(sell_df)
    feat[f"{prefix}_buy_user_num"] = buy_df.trader.nunique()
    feat[f"{prefix}_sell_user_num"] = sell_df.trader.nunique()
    feat[f"{prefix}_buy_vol"] = buy_df.value.sum()
    feat[f"{prefix}_sell_vol"] = abs(sell_df.value.sum())
    feat[f"{prefix}_avg_buy_vol"] = feat[f"{prefix}_buy_vol"] / (feat[f"{prefix}_buy_num"] + 1e-6)
    feat[f"{prefix}_avg_sell_vol"] = feat[f"{prefix}_sell_vol"] / (feat[f"{prefix}_sell_num"] + 1e-6)
    feat[f"{prefix}_sell_pressure"] = feat[f"{prefix}_sell_vol"] / (feat[f"{prefix}_buy_vol"] + 1e-6)

    # feat["flip_ratio"] = flipped_users / (feat["buy_user_num"] + 1e-6)
    # use bundle to reveal the real information
    # bundle_flag = True
    # bundle_flag = False
    # if bundle_flag:
    # ===========Feature Group 4: Bundle Statistics============
    prefix = "group4"
    uf = UnionFind()
    n = len(cluster_set_list)
    for i in range(n):
        for j in range(i + 1, n):
            if cluster_set_list[i] & cluster_set_list[j]:  # 有交集
                uf.union(i, j)
    root_to_set = defaultdict(set)
    for i, s in enumerate(cluster_set_list):
        root = uf.find(i)
        root_to_set[root].update(s)
    merged_cluster_set_list = list(root_to_set.values())
    addr2cluster = {}
    for i, cluster in enumerate(merged_cluster_set_list, start=1):
        cluster_name = f"cluster_{i}"
        for addr in cluster:
            addr2cluster[addr] = cluster_name

    hold_df["cluster"] = hold_df["trader"].map(addr2cluster)
    hold_df["cluster"] = hold_df["cluster"].fillna(hold_df["trader"])
    real_hold_df = (hold_df.groupby("cluster", as_index=False)["amount"].sum().sort_values("amount", ascending=False))
    cluster_hold_df = real_hold_df[real_hold_df["cluster"].str.startswith("cluster")]

    trader_df["cluster"] = trader_df["trader"].map(addr2cluster)
    trader_df["cluster"] = trader_df["cluster"].fillna(trader_df["trader"])

    # # holder stats
    swap_df = trader_df[trader_df['type'].str.contains('swap', case=False, na=False)]
    buy_df = swap_df[swap_df['value'] > 0].copy()
    sell_df = swap_df[swap_df['value'] < 0].copy()
    first_buyers = buy_df.drop_duplicates("cluster", keep="first")
    early_top1_buyers = first_buyers.head(1).cluster.tolist()
    early_top5_buyers = first_buyers.head(5).cluster.tolist()
    early_top10_buyers = first_buyers.head(10).cluster.tolist()
    early_top20_buyers = first_buyers.head(20).cluster.tolist()
    early_top1_holdings = hold_df[hold_df.cluster.isin(early_top1_buyers)]
    early_top5_holdings = hold_df[hold_df.cluster.isin(early_top5_buyers)]
    early_top10_holdings = hold_df[hold_df.cluster.isin(early_top10_buyers)]
    early_top20_holdings = hold_df[hold_df.cluster.isin(early_top20_buyers)]
    # early_top50_buyers = first_buyers.head(50).cluster.tolist()
    # early_top50_holdings = hold_df[hold_df.cluster.isin(early_top50_buyers)]

    feat[f"{prefix}_holder_gini"] = gini(real_hold_df[real_hold_df.amount >= 1].amount.values)
    feat[f"{prefix}_top1_pct"] = real_hold_df.head(1).amount.sum() / total_supply
    feat[f"{prefix}_top5_pct"] = real_hold_df.head(5).amount.sum() / total_supply
    feat[f"{prefix}_top10_pct"] = real_hold_df.head(10).amount.sum() / total_supply
    feat[f"{prefix}_top20_pct"] = real_hold_df.head(20).amount.sum() / total_supply
    feat[f"{prefix}_top50_pct"] = real_hold_df.head(50).amount.sum() / total_supply
    feat[f"{prefix}_top100_pct"] = real_hold_df.head(100).amount.sum() / total_supply
    feat[f"{prefix}_top10_to_top100"] = feat[f"{prefix}_top10_pct"] / (feat[f"{prefix}_top100_pct"] + 1e-6)
    feat[f"{prefix}_holder_num"] = (real_hold_df.amount >= 1).sum()
    feat[f"{prefix}_buy_user_num"] = buy_df.cluster.nunique()
    feat[f"{prefix}_sell_user_num"] = sell_df.cluster.nunique()
    feat[f"{prefix}_early_top1_hold_pct"] = early_top1_holdings.amount.sum() / total_supply
    feat[f"{prefix}_early_top1_initial_hold_pct"] = first_buyers.head(1).value.sum() / total_supply
    feat[f"{prefix}_early_top1_hold_ratio"] = feat[f"{prefix}_early_top1_hold_pct"] / feat[f"{prefix}_early_top1_initial_hold_pct"]
    feat[f"{prefix}_early_top5_hold_pct"] = early_top5_holdings.amount.sum() / total_supply
    feat[f"{prefix}_early_top5_initial_hold_pct"] = first_buyers.head(5).value.sum() / total_supply
    feat[f"{prefix}_early_top5_hold_ratio"] = feat[f"{prefix}_early_top5_hold_pct"] / feat[f"{prefix}_early_top5_initial_hold_pct"]
    feat[f"{prefix}_early_top10_hold_pct"] = early_top10_holdings.amount.sum() / total_supply
    feat[f"{prefix}_early_top10_initial_hold_pct"] = first_buyers.head(10).value.sum() / total_supply
    feat[f"{prefix}_early_top10_hold_ratio"] = feat[f"{prefix}_early_top10_hold_pct"] / feat[f"{prefix}_early_top10_initial_hold_pct"]
    feat[f"{prefix}_early_top20_hold_pct"] = early_top20_holdings.amount.sum() / total_supply
    feat[f"{prefix}_early_top20_initial_hold_pct"] = first_buyers.head(20).value.sum() / total_supply
    feat[f"{prefix}_early_top20_hold_ratio"] = feat[f"{prefix}_early_top20_hold_pct"] / feat[f"{prefix}_early_top20_initial_hold_pct"]
    feat[f"{prefix}_cluster_num"] = len(cluster_hold_df)
    feat[f"{prefix}_cluster_holder_num"] = len(hold_df[hold_df.cluster.str.startswith("cluster")])
    feat[f"{prefix}_cluster_holder_ratio"] = len(hold_df[hold_df.cluster.str.startswith("cluster")])/len(hold_df)

    feat[f"{prefix}_cluster_total_pct"] = nz(cluster_hold_df.amount.sum(skipna=True) / total_supply)
    feat[f"{prefix}_cluster_avg_pct"] = nz(cluster_hold_df.amount.mean(skipna=True) / total_supply)
    feat[f"{prefix}_cluster_max_pct"] = nz(cluster_hold_df.amount.max(skipna=True) / total_supply)

    feat[f"{prefix}_top1_pct_delta"] = feat[f"{prefix}_top1_pct"] - feat[f"group2_top1_pct"]
    feat[f"{prefix}_top5_pct_delta"] = feat[f"{prefix}_top5_pct"] - feat[f"group2_top5_pct"]
    feat[f"{prefix}_top10_pct_delta"] = feat[f"{prefix}_top10_pct"] - feat[f"group2_top10_pct"]
    feat[f"{prefix}_top20_pct_delta"] = feat[f"{prefix}_top20_pct"] - feat[f"group2_top20_pct"]
    feat[f"{prefix}_top50_pct_delta"] = feat[f"{prefix}_top50_pct"] - feat[f"group2_top50_pct"]
    feat[f"{prefix}_top100_pct_delta"] = feat[f"{prefix}_top100_pct"] - feat[f"group2_top100_pct"]

    # time-series
    try:
        df = pd.DataFrame(valid_tx_list)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df['sol_amount'] = df['sol_amount'].abs()
        df['token_amount'] = df['token_amount'].abs()
        df['volume'] = df['token_amount']
        df['price'] = df['price'].abs()

        df = df.set_index('datetime').sort_index()
        t0 = df.index.min()
        # t1 = t0 + pd.Timedelta(seconds=3600 - 1)
        t1 = df.index.max()
        print(t1-t0)

        df = df[(df.index >= t0) & (df.index <= t1)]
        agg = df.resample('10s').agg({
            'price': ['first', 'max', 'min', 'last'],
            'volume': 'sum'
        })
        agg.columns = ['open', 'high', 'low', 'close', 'volume']
        # full_index = pd.date_range(start=t0, end=t1, freq='10s')
        # agg = agg.reindex(full_index)

        if pd.isna(agg['close'].iloc[0]):
            agg['close'].iloc[0] = df['price'].iloc[0]  # 或者其他你认为合适的初始值
        agg['close'] = agg['close'].ffill()

        for col in ['open', 'high', 'low']:
            agg[col] = agg[col].fillna(agg['close'])

        agg['volume'] = agg['volume'].fillna(0)

        ######################
        MAX_T = 360
        feat_cols = ['open', 'high', 'low', 'close', 'volume']
        ts = agg[feat_cols].to_numpy(dtype='float32')  # shape: (L, 5)
        L, D = ts.shape
        if L >= MAX_T:
            ts_fixed = ts[-MAX_T:]
            seq_len = MAX_T
        else:
            pad_len = MAX_T - L
            pad = np.zeros((pad_len, D), dtype='float32')
            if L > 0:
                pad[:, 0:4] = ts[-1, 0:4]  # open/high/low/close 用最后值
                # pad[:, 4] = 0  # volume 已经是 0，不写也行
            ts_fixed = np.concatenate([ts, pad], axis=0)
            seq_len = L

        feat["ts"] = ts_fixed
        feat["ts_len"] = seq_len
    except:
        print(mint_address)

    return feat

if __name__ == '__main__':

    price_dict = {}
    with open("../data/sol_hourly.txt", "r") as f:
        for line in f:
            ts_str, price_str = line.rstrip().split("\t")
            price_dict[int(ts_str)] = float(price_str)

    est = pytz.timezone("America/New_York")
    feat_list = []
    random.seed(42)
    cnt = 0

    # Jito bundle information
    with open("../data/bundle2sig_filtered.pkl", "rb") as f:
        bundle2sig_filtered = pkl.load(f)
    # sig to bundle
    sig2bundle = {}
    for bundle_id, sigs in bundle2sig_filtered.items():
        for sig in sigs:
            sig2bundle[sig] = bundle_id

    exchange_signer = set()
    with open("../data/exchange_account.txt", "r") as f:
        for line in f.readlines():
            exchange_signer.add(line[:-1])

    account2signer = {}
    with open("../data/first_sig_signer.jsonl", "r") as f:
        for line in f.readlines():
            info = json.loads(line)
            account = info["account"]
            signer = info["signer"]
            if account and signer and (signer not in exchange_signer):
                account2signer[account] = signer

    signer2tx_list = {}
    dir_name = "../raw_data/inner_tx/memefull"
    file_list = os.listdir(dir_name)
    exp_token_list = []

    # small dataset for development
    for token_tx_file in tqdm(file_list):
        token_address = token_tx_file.split(".")[0]
        tx_list = []
        with open(f"{dir_name}/{token_tx_file}", "r") as f:
            for line in f:
                tx = json.loads(line)
                tx_list.append(tx)
        # some of the token's label are not included
        feat = feature_generation(token_address, tx_list, price_dict, sig2bundle, account2signer)
        if feat:
            feat_list.append(feat)

    X_df = pd.DataFrame(feat_list)
    X_df.to_csv("../data/feat_with_ts.csv", index=False)

    with open("../data/feat_with_ts.pkl", "wb") as f:
        pkl.dump(feat_list, f, protocol=pkl.HIGHEST_PROTOCOL)

