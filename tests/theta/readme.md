# uniswap v3 theta计算

## 目的

计算uniswap v3上每天的手续费收益. 作为theta的计算依据

## 计算步骤

1. 获取每天的swap交易
2. 根据swap交易的金额. 进行手续费分成. 
3. 将手续费分成累计, 作为当天的手续费收益. 
4. 用前一天的手续费收益, 预估今天手续费收益的值.

## 手续费分成的计算逻辑

手续费分成模型: 假定投入了资金为N, tick范围为(a,b), 那么如果当当前swap交易的tick在(a,b)之间的时候, 计算手续费(如果在区间之外计算为0)

计算公式为: (虚拟资金的liquidity/总流动性) * 手续费费率 * swap的金额

## 入口函数说明

入口函数在core.py中. 

### get_uniswap_theta:

根据给定的一天的uniswap交易数据, 以及特定的做市范围, 计算手续费收入. 注意: 返回的手续费收入是一个比值, 也就是手续费收入的金额/虚拟资金的金额 

```python
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
    pass
```

### get_uniswap_theta_grid: 

估算在tick区间中, 各种tick组合的手续费收益, 具体来说, 根据给定期望的tick范围, 和分布情况, 生成一个不同tick区间的网格. 然后根据给定的一天的uniswap的交易数据, 计算网格中每个点的手续费情况, 函数原型为: 

```python
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
    :return: 每种tick组合, 以及对应的手续费
    :rtype:
    """
    pass
```

lower_rate,upper_rate: 这是网格的最大范围和最小范围的比值, 比如0.5和1.5

lower_interval,upper_interval: 控制网格的密度,

比如当密度=0.1, power=1的时候, 会生成0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5的序列

当密度=0.2, power=1的时候, 会生成0.5,0.7,0.9,1.1,1.3,1.5的序列

power: 为了让数据点在1中心密集, 在两边稀疏, 支持通过设定不同的指数, 让生成的序列弯曲一些. 生成的过程中, 首先根据范围和interval生成均匀的序列, 如0.5,0.7,0.9,1.1,1.3,1.5, 

然后对这些数据折算到0~1的区间, 然后取指数, power就是指数多少. 

比如对于0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5, 当power=2, 序列为0.5, 0,68, 0.82, 0.92, 0.98, 1, 1.02, 1.08, 1.18, 1.32, 1.5, 可见在1周围更密集, 在两头更稀疏, 当power=1的时候, 相当于生成一个等差数列

然后会将刚才得到的序列, 用每个点做排列组合, 并将每种组合都作为做市范围, 然后计算手续费收益. (注意这些做市范围必须包含1, 不包括1的都被丢弃了, 比如0.5 ~ 1.1是有效的, 0.5 ~ 0.8就是无效的)

### get_uniswap_theta_grid_batch:

同上, 但是支持一下读取一个文件夹里所有的数据计算.

每个文件是一天的交易. 会自动使用当天最后一个swap的price作为第二天做市的中心价格. 



# 使用例子参见:

[theta_test.py](..%2Ftests%2Ftheta_test.py) 中test_get_fee_amount函数.

其中的csv文件, 是通过demeter下载的tick级别的数据. 
