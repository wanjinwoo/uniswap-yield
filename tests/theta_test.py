import unittest
from decimal import Decimal
import pandas as pd
import theta.core as tc
from theta.helper import tick_to_sqrtPriceX96, sqrt_price_to_tick, tick_to_quote_price, sqrtPriceX96_to_quote_price
from theta.liquitidy_math import get_liquidity_for_amount0, get_liquidity_for_amount1, get_liquidity


class TestTheta(unittest.TestCase):

    #
    def test_generate_grid(self):
        print(tc.generate_grid(tc.PoolInfo(True, 6, 18, Decimal(0.3)),
                               Decimal("0.5"),
                               Decimal("1.5"),
                               Decimal("1000"),
                               lower_interval=Decimal(0.1),
                               upper_interval=Decimal(0.1),
                               power=Decimal(2)))

    def test_get_amount01(self):
        tick_lower = 200000
        sqrtA = tick_to_sqrtPriceX96(tick_lower)
        tick_upper = 300000
        sqrtB = tick_to_sqrtPriceX96(tick_upper)
        amount = 1000
        a0 = get_liquidity_for_amount0(sqrtA, sqrtB, amount * 10 ** 6)
        a1 = get_liquidity_for_amount1(sqrtA, sqrtB, amount * 10 ** 6)
        a = get_liquidity(sqrtA, tick_lower, tick_upper, amount, 1, 6, 18)
        print(a0, a1, a)

    def test_get_price(self):
        print(sqrtPriceX96_to_quote_price(130720691992206303120161, 18, 6, False))

    def test_get_fee_amount(self):
        result = tc.get_uniswap_theta_grid(
            "/home/sun/AA-labs-FE/code/research/data_tick/0x45dda9cb7c25131df268515131f647d726f50608-2022-01-05.csv",
            tc.PoolInfo(True, 6, 18, Decimal(0.05), base_amount=Decimal(1000)),
            Decimal("0.95"),
            Decimal("1.05"),
            Decimal("3504"),
        )
        # result.to_csv("amount1.csv")
        print(result)

    def test_get_fee_amount(self):
        result = tc.get_uniswap_theta_grid(
            "/home/sun/AA-labs-FE/code/research/data_tick/0x45dda9cb7c25131df268515131f647d726f50608-2022-01-05.csv",
            tc.PoolInfo(True, 6, 18, Decimal(0.05), base_amount=Decimal(1000)),
            Decimal("0.9"),
            Decimal("1.1"),
            Decimal("3504"),
            upper_interval=Decimal("0.01"),
            lower_interval=Decimal("0.01"),
            power=Decimal(1)
        )
        # result.to_csv("amount1.csv")
        print(result)

    def test_get_fee_amount_1_base(self):
        result = tc.get_uniswap_theta_grid(
            "/home/sun/AA-labs-FE/05_op_reward_phase2/data-tick/0x1c3140ab59d6caf9fa7459c6f83d4b52ba881d36-2023-02-02.csv",
            tc.PoolInfo(False, 18, 6, Decimal(0.3), base_amount=Decimal(1000)),
            Decimal("0.95"),
            Decimal("1.05"),
            Decimal("2.722259417684684658106734776"),

        )
        # result.to_csv("amount1.csv")
        print(result)

    def test_get_fee_amount_wide(self):
        result = tc.get_uniswap_theta_grid(
            "/home/sun/AA-labs-FE/code/research/data_tick/0x45dda9cb7c25131df268515131f647d726f50608-2022-01-05.csv",
            tc.PoolInfo(True, 6, 18, Decimal(0.05), base_amount=Decimal(1000)),
            Decimal("0.01"),
            Decimal("1.99"),
            Decimal("3504"), 
        )
        # result.to_csv("amount1000_w.csv")
        print(result)

    def test_get_fee_amount_2(self):
        result = tc.get_uniswap_theta_grid(
            "./0x45dda9cb7c25131df268515131f647d726f50608-2023-03-16.csv",
            tc.PoolInfo(True, 6, 18, Decimal(0.05), base_amount=Decimal(1000)),
            Decimal("0.01"),
            Decimal("1.99"),
            Decimal("1650"),
        )
        result.to_csv("matic_usdc_weth_2023-03-16.csv")
        # print(result)

    def test_current_tick_range(self):
        df: pd.DataFrame = pd.read_csv("./0x45dda9cb7c25131df268515131f647d726f50608-2023-03-16.csv")
        print(df.loc[(df.current_tick >= 201722) & (df.current_tick <= 201930)])

    def test_get_uniswap_theta_batch(self):
        tc.get_uniswap_theta_grid_batch("/home/sun/AA-labs-FE/code/uniswap-lib/toys/计算一个时期的theta",
                                        "/home/sun/AA-labs-FE/code/uniswap-lib/toys/计算一个时期的theta/result",
                                        tc.PoolInfo(True, 6, 18, Decimal(0.05), base_amount=Decimal(1000)),
                                        Decimal("0.2"),
                                        Decimal("5"),
                                        lower_interval=Decimal("0.04"),
                                        upper_interval=Decimal("0.2"),
                                        power=Decimal(2)
                                        )

    def test_get_line(self):
        one = Decimal(1)
        print(tc.get_square_range(Decimal(0.5), Decimal(1), Decimal(0.1), one))
        print(tc.get_square_range(Decimal(0.5), Decimal(2.5), Decimal(0.1), one))
        print(tc.get_square_range(Decimal(1), Decimal(3), Decimal(0.1), one))
        print(tc.get_square_range(Decimal(1), Decimal(1.5), Decimal(0.05), one))

    def test_get_line_neg(self):
        print(tc.get_square_range_neg(Decimal("0.2"), Decimal("1"), Decimal("0.04"), Decimal(2)))
        print(tc.get_square_range(Decimal("1"), Decimal("5"), Decimal("0.2"), Decimal(2)))

    def test_get_single_point(self):
        value = tc.get_uniswap_theta("/home/sun/AA-labs-FE/05_op_reward_phase2/data-tick/0x1c3140ab59d6caf9fa7459c6f83d4b52ba881d36-2023-02-02.csv",
                                     tc.PoolInfo(False, 18, 6, Decimal(0.3), base_amount=Decimal(1000)),
                                     Decimal("0.95"),
                                     Decimal("1.05"),
                                     Decimal("2.722259417684684658106734776"))
        print(value)


    
