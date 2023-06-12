import os
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import NamedTuple, Dict
from .helper import quote_price_to_tick, tick_to_sqrtPriceX96, quote_price_to_tick_nobase, sqrtPriceX96_to_quote_price
import pandas as pd
import math
from tqdm import tqdm  # process bar
from .liquitidy_math import get_liquidity, get_liquidity_for_amount0
import numpy as np
import re


@dataclass
class PoolInfo:
    is_0_base: bool
    token0_decimal: int
    token1_decimal: int
    fee_rate: Decimal
    base_amount: Decimal = Decimal("1000")


def get_square_range(start, end, interval, power):
    """
    把数字序列, 从等差序列变化为符合 x^2(0~1)区间的分布
    先生成0~1区间的序列, 然后x**2, 最后做尺度变换, 调整到输入参数的区间.
    比如: 输入1~1.5, 间隔0.1,
    等差序列是1,1.1,1.2,1.3,1.4,1,5
    本函数结果为1,1.02,1.08,1.18,1.32,1.5

    目的是使得结果序列在start这边密集一些, end这边稀疏一些
    power是调整曲线斜率的, 等于1的时候就是等差数列, 越大在start这边就越密集.
    """

    amp = (end - start)
    return (np.arange(Decimal(0), Decimal(1.00001), interval / amp) ** power) * amp + start


def get_square_range_neg(start, end, interval, power):
    """
    把数字序列, 从等差序列变化为x^2(-1~0)区间的分布
    比如: 输入0.5 ~ 1, 间隔0.1, 结果为0.5,0.68,0.82,0.92,0.98,1
    目的是使得结果序列在start这边稀疏一些, end这边密集一些
    """

    amp = (end - start)
    return np.flip(1 - (np.arange(Decimal(0), Decimal(1.00001), interval / amp) ** power)) * amp + start


def generate_grid(pool_info: PoolInfo,
                  lower_rate: Decimal,
                  upper_rate: Decimal,
                  price: Decimal,
                  lower_interval=Decimal("0.01"),
                  upper_interval=Decimal("0.01"),
                  power=Decimal(1)):
    if not lower_rate < 1 < upper_rate:
        raise RuntimeError("must lower_rate<1<upper_rate")
    rate_array1 = get_square_range_neg(lower_rate, 1, lower_interval, power)
    rate_array2 = get_square_range(1, upper_rate, upper_interval, power)
    rate_array = np.unique(np.concatenate((rate_array1, rate_array2)))

    if pool_info.is_0_base:
        price = 1 / price
    price_array = list(map(lambda r: r * price, rate_array))
    tick_array = list(map(lambda p: quote_price_to_tick_nobase(p,
                                                               pool_info.token0_decimal,
                                                               pool_info.token1_decimal), price_array))
    tick_price_dict = dict(zip(tick_array, rate_array))

    upper_list = []
    lower_list = []
    for i in range(len(tick_array)):
        for j in range(len(tick_array) - i):
            if j == 0:
                continue
            if tick_price_dict[tick_array[i]] <= Decimal(1) <= tick_price_dict[tick_array[i + j]]:  # 只保留跨中心的
                lower_list.append(tick_array[i])
                upper_list.append(tick_array[i + j])
    return upper_list, lower_list, tick_price_dict


def get_uniswap_theta_grid_batch(from_dir: str,
                                 to_dir: str,
                                 pool_info: PoolInfo,
                                 lower_rate: Decimal,
                                 upper_rate: Decimal,
                                 lower_interval=Decimal("0.01"),
                                 upper_interval=Decimal("0.01"),
                                 power=Decimal(1)):
    files = os.listdir(from_dir)
    files = list(filter(lambda x: os.path.splitext(x)[1] == ".csv", files))
    for day_file in files:
        df: pd.DataFrame = pd.read_csv(os.path.join(from_dir, day_file))
        date_str = re.findall("\\d{4}-\\d{2}-\\d{2}", day_file)[0]
        print(date_str)
        first_sqrt_x96 = int(df[df.tx_type == "SWAP"].sqrtPriceX96.tail(1))  # 当天最后一次成交的价格
        price = sqrtPriceX96_to_quote_price(first_sqrt_x96, pool_info.token0_decimal, pool_info.token1_decimal, pool_info.is_0_base)

        df = get_uniswap_theta_grid(os.path.join(from_dir, day_file),
                                    pool_info,
                                    lower_rate,
                                    upper_rate,
                                    price,
                                    lower_interval,
                                    upper_interval,
                                    power)
        df.to_csv(os.path.join(to_dir, date_str + ".csv"), index=False)
        pass


def get_uniswap_theta_grid(tx_file: str,
                           pool_info: PoolInfo,
                           lower_rate: Decimal,
                           upper_rate: Decimal,
                           price: Decimal,
                           lower_interval=Decimal("0.01"),
                           upper_interval=Decimal("0.01"),
                           power=Decimal(1)
                           ) -> pd.DataFrame:
    """
    估算在tick区间中, 各种tick组合的手续费收益

    网格相关的参数包括
    lower_rate,upper_rate: 这是网格的最大范围和最小范围的比值, 比如0.5和1.5
    lower_interval,upper_interval: 控制网格的密度,
        比如当密度=0.1, power=1的时候, 会生成0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5的序列
        当密度=0.2, power=1的时候, 会生成0.5,0.7,0.9,1.1,1.3,1.5的序列
    power: 为了让数据点在1中心密集, 在两边稀疏, 支持通过设定不同的指数, 让生成的序列弯曲一些. 生成的过程中, 首先根据范围和interval生成均匀的序列, 如0.5,0.7,0.9,1.1,1.3,1.5,
           然后对这些数据折算到0~1的区间, 然后取指数, power就是指数多少.
           比如对于0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5, 当power=2, 序列为0.5, 0,68, 0.82, 0.92, 0.98, 1, 1.02, 1.08, 1.18, 1.32, 1.5, 可见在1周围更密集, 在两头更稀疏

    然后会将刚才得到的序列, 用每个点做排列组合, 生成一个网格. 并将每种组合都作为做市范围, 然后计算手续费收益. (注意这些做市范围必须包含1, 不包括1的都被丢弃了, 比如0.5~1.1是有效的, 0.5~0.8就是无效的)

    :param tx_file: 用于估算的文件, 内容是前一天所有交易的数据. 只用到其中swap的部分
    :type tx_file:
    :param pool_info: 池子信息
    :type pool_info:
    :param lower_rate: 最小做市范围, 与中心价格的比值, 必须<1,
    :type lower_rate:
    :param upper_rate: 最大做市范围, 与中心价格的比值, 必须>1
    :type upper_rate:
    :param price: 中心价格
    :type price:
    :param lower_interval: 生成网格的时候, <1部分tick的步长
    :type upper_interval:
    :param upper_interval: 生成网格的时候, >1部分tick的步长
    :type upper_interval:
    :param power: 生成网格时候, 步长的调整参数
    :type power:
    :return:
    :rtype:
    """
    setattr(pool_info, "base_decimal", pool_info.token0_decimal if pool_info.is_0_base else pool_info.token1_decimal)
    setattr(pool_info, "real_fee_rate", Decimal(pool_info.fee_rate / 100))
    if lower_rate >= 1 or upper_rate <= 1:
        raise RuntimeError("lower should < 1 and upper should > 1")
    amount_column = "amount" + ("0" if pool_info.is_0_base else "1")
    df: pd.DataFrame = pd.read_csv(tx_file, dtype={'total_liquidity': 'float'})
    df = df.loc[df.tx_type == "SWAP"]
    df = df[[amount_column, "current_tick", "total_liquidity"]]
    df = df.rename(columns={amount_column: "amount"})

    upper_list, lower_list, tick_price_dict = generate_grid(pool_info, lower_rate, upper_rate, price, lower_interval, upper_interval, power)
    fee_rate_list = []
    with tqdm(total=len(upper_list), ncols=150) as pbar:
        for i in range(len(upper_list)):
            fee_rate = __calc_one_tick(df, lower_list[i], pool_info, upper_list[i])
            fee_rate_list.append(fee_rate)
            pbar.update()
    upper_rate_list = map(lambda x: tick_price_dict[x], upper_list)
    lower_rate_list = map(lambda x: tick_price_dict[x], lower_list)
    result_df = pd.DataFrame({
        "lower_rate": lower_rate_list, "upper_rate": upper_rate_list, "fee_rate": fee_rate_list
    })
    return result_df


def get_uniswap_theta(tx_file: str,
                      pool_info: PoolInfo,
                      lower: Decimal,
                      upper: Decimal,
                      price: Decimal
                      ) -> Decimal:
    """
    根据给定的一天的uniswap交易数据, 以及特定的做市范围, 计算手续费收入. 注意: 返回的手续费收入是一个比值, 也就是手续费收入的金额/虚拟资金的金额
    :param tx_file: 一天的swap交易数据,
    :type tx_file:
    :param pool_info: 池子的信息, 包括base_token, decimal等
    :type pool_info:
    :param lower:  期望做市范围的下限, 与中心价格的比值. 必须小于1, 比如0.5, 那么当price=1000的时候, 对应的价格是500
    :type lower:
    :param upper: 期望做市范围的上限, 与中心价格的比值. 必须大于1, 比如1.5, 那么当price=1000的时候, 对应的价格是1500
    :type upper:
    :param price: 中心价格.
    :type price:
    :return: 返回值是手续费收益的比值, 假设本金是1000(这个参数是内置的), 手续费收益是20, 那么返回值是20/1000=0.02
    :rtype:
    """
    setattr(pool_info, "base_decimal", pool_info.token0_decimal if pool_info.is_0_base else pool_info.token1_decimal)
    setattr(pool_info, "real_fee_rate", Decimal(pool_info.fee_rate / 100))
    if lower >= 1 or upper <= 1:
        raise RuntimeError("lower should < 1 and upper should > 1")
    amount_column = "amount" + ("0" if pool_info.is_0_base else "1")
    df: pd.DataFrame = pd.read_csv(tx_file, dtype={'total_liquidity': 'float'})
    df = df.loc[df.tx_type == "SWAP"]
    df = df[[amount_column, "current_tick", "total_liquidity"]]
    df = df.rename(columns={amount_column: "amount"})

    if pool_info.is_0_base:
        price = 1 / price
    lower_tick = quote_price_to_tick_nobase(price * lower, pool_info.token0_decimal, pool_info.token1_decimal)
    upper_tick = quote_price_to_tick_nobase(price * upper, pool_info.token0_decimal, pool_info.token1_decimal)

    fee_rate = __calc_one_tick(df, lower_tick, pool_info, upper_tick)

    return fee_rate


def get_real_daily_fee(from_dir: str,
                    pool_info: PoolInfo,
                    ) -> pd.DataFrame:
    
    setattr(pool_info, "base_decimal", pool_info.token0_decimal if pool_info.is_0_base else pool_info.token1_decimal)
    setattr(pool_info, "real_fee_rate", Decimal(pool_info.fee_rate / 100))

    files = os.listdir(from_dir)
    files = list(filter(lambda x: x.endswith(".tick.csv"), files))

    date_list = []
    lower_price_list = []
    upper_price_list = []
    fee_rate_list = []

    for day_file in files:
        df: pd.DataFrame = pd.read_csv(os.path.join(from_dir, day_file))
        date_str = re.findall("\\d{4}-\\d{2}-\\d{2}", day_file)[0]
        print(date_str)
    

        lower_sqrt_x96 = int(df[df.tx_type == "SWAP"].sqrtPriceX96.min) # 当天最低价格
        lower_tick = int(df[df.tx_type == "SWAP"].current_tick.min)
        lower_price = sqrtPriceX96_to_quote_price(lower_sqrt_x96, pool_info.token0_decimal, pool_info.token1_decimal, pool_info.is_0_base)
        upper_sqrt_x96 = int(df[df.tx_type == "SWAP"].sqrtPriceX96.max) # 当天最高价格  
        upper_tick = int(df[df.tx_type == "SWAP"].current_tick.max)
        upper_price = sqrtPriceX96_to_quote_price(upper_sqrt_x96, pool_info.token0_decimal, pool_info.token1_decimal, pool_info.is_0_base)

        fee_rate = __calc_one_tick(df, lower_tick, pool_info, upper_tick)

        date_list.append(date_str)
        lower_price_list.append(lower_price)
        upper_price_list.append(upper_price)
        fee_rate_list.append(fee_rate)

    result_df = pd.DataFrame({
                            'date': date_list,
                            'lower_price': lower_price_list,
                            'upper_price': upper_price_list,
                            'fee_rate': fee_rate_list
                            })


    return result_df

    




def __calc_one_tick(df, lower, pool_info, upper):
    if pool_info.is_0_base:
        sqrtA = tick_to_sqrtPriceX96(lower)
        liquidity = get_liquidity(sqrtA, lower, upper, pool_info.base_amount, 0,
                                  pool_info.token0_decimal, pool_info.token1_decimal)
    else:
        sqrtB = tick_to_sqrtPriceX96(upper)
        liquidity = get_liquidity(sqrtB, lower, upper, 0, pool_info.base_amount,
                                  pool_info.token0_decimal, pool_info.token1_decimal)
    fee = get_tick_fee(df, pool_info, lower, upper, liquidity)
    fee_rate = fee / pool_info.base_amount
    return fee_rate


def get_tick_fee(df: pd.DataFrame, pool_info: PoolInfo, tick_lower: int, tick_upper: int, liquidity: int) -> Decimal:
    ndf: pd.DataFrame = df[(tick_lower <= df["current_tick"]) & (df["current_tick"] <= tick_upper)]
    if len(ndf.index) > 0:
        # 手续费分成模型: 流动性占比 * 手续费费率 * base_coin的amount, 具体如下
        # liquidity/row["total_liquidity"] * pool_info.real_fee_rate *
        # Decimal(abs(int(row["amount"]))) / 10 ** pool_info.base_decimal
        # 让pandas先做列相关的计算, 然后总和拿出来计算剩下的部分, 会非常快.
        fees = ndf["amount"] / ndf["total_liquidity"]
        return Decimal(fees.abs().sum()) * pool_info.real_fee_rate * Decimal(liquidity / 10 ** pool_info.base_decimal)
    else:
        return Decimal(0)
