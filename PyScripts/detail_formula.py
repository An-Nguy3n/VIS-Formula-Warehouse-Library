import numpy as np
from numba import njit
import eval_funcs as foo
from base import Base, convert_strF_to_arrF, calculate_formula
import pandas as pd


@njit
def find_rank(A, v):
    left = 0
    right = len(A) - 1
    while left <= right:
        mid = (left + right) // 2

        if A[mid] > v:
            left = mid + 1
        else:
            right = mid - 1

    return left + 1


def singleCompanyInvest(vis: Base, vis1: Base, weight, weight1):
    GeoMax, HarMax, ValGeo, GeoLim, ValHar, HarLim, GeoRank, HarRank = foo.singleCompanyInvest(
        weight, vis.INDEX, vis.PROFIT, vis.PROFIT_RANK, vis.PROFIT_RANK_NI, vis.INTEREST
    )

    list_invest, list_profit = foo.singleCompanyInvest_test(
        weight1, vis1.INDEX, vis1.PROFIT, vis1.INTEREST
    )

    if list_invest[0] == -1:
        CtyMax = "NotInvest"
    else:
        CtyMax = vis1.data.loc[list_invest[0], "SYMBOL"]

    ProMax = list_profit[0]
    return GeoMax, HarMax, CtyMax, ProMax, ValGeo, GeoLim, ValHar, HarLim, GeoRank, HarRank


def singleYearThreshold(vis: Base, weight):
    ValGeoNgn, GeoNgn, ValHarNgn, HarNgn = foo.singleYearThreshold(
        weight, vis.INDEX, vis.PROFIT, vis.INTEREST
    )

    list_invest, list_profit = foo.singleYearThreshold_test(
        weight, vis.INDEX, vis.PROFIT, vis.INTEREST, ValHarNgn
    )

    ProNgn = list_profit[-1]
    CtyNgn = "_".join([vis.data.loc[k, "SYMBOL"] for k in list_invest[-1]])

    list_profit_test = list_profit[:-1]
    list_profit_rank = []
    for i in range(len(list_profit_test)):
        list_profit_rank.append(find_rank(
            vis.sorted_PROFIT[-i-1], list_profit_test[i]
        ))

    list_profit_rank = np.array(list_profit_rank, float)
    RankGeo1 = foo.geomean(list_profit_rank)
    RankHar1 = foo.harmean(list_profit_rank)

    return ValGeoNgn, GeoNgn, ValHarNgn, HarNgn, CtyNgn, ProNgn, RankGeo1, RankHar1


def doubleYearThreshold(vis: Base, weight):
    ValGeoNgn2, GeoNgn2, ValHarNgn2, HarNgn2, last_reason = foo.doubleYearThreshold(
        weight, vis.INDEX, vis.PROFIT, vis.SYMBOL, vis.INTEREST, vis.BOOL_ARG
    )

    list_invest, list_profit = foo.doubleYearThreshold_test(
        weight, vis.INDEX, vis.PROFIT, vis.SYMBOL, vis.INTEREST, vis.BOOL_ARG, ValHarNgn2, 0
    )

    ProNgn2 = list_profit[-1]
    CtyNgn2 = "_".join([vis.data.loc[k, "SYMBOL"] for k in list_invest[-1]])

    list_profit_test = list_profit[:-1]
    list_profit_rank = []
    for i in range(len(list_profit_test)):
        list_profit_rank.append(find_rank(
            vis.sorted_PROFIT[-i-2], list_profit_test[i]
        ))

    list_profit_rank = np.array(list_profit_rank, float)
    RankGeo2 = foo.geomean(list_profit_rank)
    RankHar2 = foo.harmean(list_profit_rank)

    return ValGeoNgn2, GeoNgn2, ValHarNgn2, HarNgn2, CtyNgn2, ProNgn2, RankGeo2, RankHar2


def tripleYearThreshold(vis: Base, weight):
    ValGeoNgn3, GeoNgn3, ValHarNgn3, HarNgn3, last_reason = foo.tripleYearThreshold(
        weight, vis.INDEX, vis.PROFIT, vis.SYMBOL, vis.INTEREST, vis.BOOL_ARG
    )

    list_invest, list_profit = foo.tripleYearThreshold_test(
        weight, vis.INDEX, vis.PROFIT, vis.SYMBOL, vis.INTEREST, vis.BOOL_ARG, ValHarNgn3, 0
    )

    ProNgn3 = list_profit[-1]
    CtyNgn3 = "_".join([vis.data.loc[k, "SYMBOL"] for k in list_invest[-1]])

    list_profit_test = list_profit[:-1]
    list_profit_rank = []
    for i in range(len(list_profit_test)):
        list_profit_rank.append(find_rank(
            vis.sorted_PROFIT[-i-3], list_profit_test[i]
        ))

    list_profit_rank = np.array(list_profit_rank, float)
    RankGeo3 = foo.geomean(list_profit_rank)
    RankHar3 = foo.harmean(list_profit_rank)

    return ValGeoNgn3, GeoNgn3, ValHarNgn3, HarNgn3, CtyNgn3, ProNgn3, RankGeo3, RankHar3


def process(data_full: pd.DataFrame, df_CT: pd.DataFrame, interest: float, valuearg_threshold: float, time: int):
    data_full = data_full.copy()
    data_full["SYMBOL"] = data_full["SYMBOL"].astype(str)

    data = data_full[data_full["TIME"]<=time].reset_index(drop=True)
    data1 = data_full[data_full["TIME"]==time].reset_index(drop=True)
    vis = Base(data, interest, valuearg_threshold)
    vis1 = Base(data1, interest, valuearg_threshold)

    all_data = []
    for i in range(len(df_CT)):
        temp_data = []
        ct = df_CT.loc[i, "CT"]
        temp_data.append(df_CT.loc[i, "id"])
        temp_data.append(ct)
        ct = convert_strF_to_arrF(ct)
        weight = calculate_formula(ct, vis.OPERAND)
        weight1 = calculate_formula(ct, vis1.OPERAND)

        temp_data.extend(list(
            singleCompanyInvest(vis, vis1, weight, weight1)
        ))

        temp_data.extend(list(
            singleYearThreshold(vis, weight)
        ))

        temp_data.extend(list(
            doubleYearThreshold(vis, weight)
        ))

        temp_data.extend(list(
            tripleYearThreshold(vis, weight)
        ))

        temp_data.extend(list(foo.find_slope(
            weight, vis.INDEX, vis.PROFIT, vis.INTEREST
        )))

        temp_data.extend([foo.getNoBalanceValue(
            weight, vis.INDEX, vis.PROFIT
        )])

        #
        all_data.append(temp_data)

    list_column = ["id", "CT"]\
                + "GeoMax, HarMax, CtyMax, ProMax, ValGeo, GeoLim, ValHar, HarLim, GeoRank, HarRank".split(", ")\
                + "ValGeo1, GeoNgn1, ValHar1, HarNgn1, CtyNgn1, ProNgn1, RankGeo1, RankHar1".split(", ")\
                + "ValGeo2, GeoNgn2, ValHar2, HarNgn2, CtyNgn2, ProNgn2, RankGeo2, RankHar2".split(", ")\
                + "ValGeo3, GeoNgn3, ValHar3, HarNgn3, CtyNgn3, ProNgn3, RankGeo3, RankHar3".split(", ")\
                + "Slope_avg, Slope_wgt_avg".split(", ")\
                + ["NoBalanceValue"]

    df_new = pd.DataFrame(all_data, columns=list_column)
    return df_new
