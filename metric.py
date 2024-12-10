import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

# def DOC(pred, true):

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def __sst(y_no_fitting):
    """
    计算SST(total sum of squares) 总平方和
    :param y_no_predicted: List[int] or array[int] 待拟合的y
    :return: 总平方和SST
    """
    y_mean = sum(y_no_fitting) / len(y_no_fitting)
    s_list =[(y - y_mean)**2 for y in y_no_fitting]
    sst = sum(s_list)
    return sst

def __ssr(y_fitting, y_no_fitting):
    """
    计算SSR(regression sum of squares) 回归平方和
    :param y_fitting: List[int] or array[int]  拟合好的y值
    :param y_no_fitting: List[int] or array[int] 待拟合y值
    :return: 回归平方和SSR
    """
    y_mean = sum(y_no_fitting) / len(y_no_fitting)
    s_list =[(y - y_mean)**2 for y in y_fitting]
    ssr = sum(s_list)
    return ssr


def __sse(y_fitting, y_no_fitting):
    """
    计算SSE(error sum of squares) 残差平方和
    :param y_fitting: List[int] or array[int] 拟合好的y值
    :param y_no_fitting: List[int] or array[int] 待拟合y值
    :return: 残差平方和SSE
    """
    s_list = [(y_fitting[i] - y_no_fitting[i])**2 for i in range(len(y_fitting))]
    sse = sum(s_list)
    return sse

def OWA(pred, true, naive_pred, train_y):
    smape = 2*np.mean(np.abs((pred - true) / (np.abs(true)+np.abs(pred))))
    mase = np.mean(np.abs(pred-true))/np.mean(np.abs(np.diff(train_y)))
    smape_n = 2*np.mean(np.abs((naive_pred - true) / (np.abs(true)+np.abs(naive_pred))))
    mase_n = np.mean(np.abs(naive_pred-true))/np.mean(np.abs(np.diff(train_y)))
    owa = (smape/smape_n+mase/mase_n)/2
    return owa


def R2(y_obs, y_sim):
    # 观测值的平均值, 对应(LE, E)
    y_obs_mean = y_obs.mean()
    y_sim_mean = y_sim.mean()
    sum1 = 0
    sum2 = 0
    sum3 = 0
    for i in range(len(y_obs)):
        sum1 = sum1 + (y_sim[i] - y_sim_mean) * (y_obs[i] - y_obs_mean)
        sum2 = sum2 + ((y_sim[i] - y_sim_mean) ** 2)
        sum3 = sum3 + ((y_obs[i] - y_obs_mean) ** 2)
    R2 = float(sum1 / ((sum2 ** 0.5) * (sum3 ** 0.5))) ** 2

    return R2

def metric(pred, true):
    pred, true = pred.reshape(-1,1),true.reshape(-1,1)
    #以防进来千奇百怪开始广播....
    mae = MAE(pred, true)
    # mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    # mspe = MSPE(pred, true)
    # rse = RSE(pred, true)
    # rr = R2(pred,true)
    # corr = CORR(pred, true)
    res = [mae, rmse, mape]

    return res
