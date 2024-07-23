import pandas as pd
from base import convert_strF_to_arrF, calculate_formula, Base
import eval_funcs as funcs
import numpy as np
from numba import njit


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


def process(data_full: pd.DataFrame, df_CT: pd.DataFrame, interest: float, valuearg_threshold: float, time: int):
    data_full = data_full.copy()
    data_full["SYMBOL"] = data_full["SYMBOL"].astype(str)

    data = data_full[data_full["TIME"]<=time].reset_index(drop=True)
    data1 = data_full[data_full["TIME"]==time].reset_index(drop=True)
    data2 = data_full[data_full["TIME"]>=time-1].reset_index(drop=True)
    data3 = data_full[data_full["TIME"]>=time-2].reset_index(drop=True)
    vis = Base(data, interest, valuearg_threshold)
    vis1 = Base(data1, interest, valuearg_threshold)
    vis2 = Base(data2, interest, valuearg_threshold)
    vis3 = Base(data3, interest, valuearg_threshold)

    all_data = []
    for i in range(len(df_CT)):
        temp_data = []
        ct = df_CT.loc[i, "CT"]
        temp_data.append(df_CT.loc[i, "id"])
        temp_data.append(ct)
        ct = convert_strF_to_arrF(ct)
        weight = calculate_formula(ct, vis.OPERAND)
        weight1 = calculate_formula(ct, vis1.OPERAND)
        weight2 = calculate_formula(ct, vis2.OPERAND)
        weight3 = calculate_formula(ct, vis3.OPERAND)

        # Single invest
        GeoMax, HarMax, ValGeo, GeoLim, ValHar, HarLim, GeoRank, HarRank = funcs.singleCompanyInvest(
            weight, vis.INDEX, vis.PROFIT, vis.PROFIT_RANK, vis.PROFIT_RANK_NI, vis.INTEREST
        )

        list_invest, list_profit = funcs.singleCompanyInvest_test(
            weight1, vis1.INDEX, vis1.PROFIT, vis1.INTEREST
        )
        if list_invest[0] == -1:
            CtyMax = "NotInvest"
        else:
            CtyMax = vis1.data.loc[list_invest[0], "SYMBOL"]

        ProMax = list_profit[0]
        temp_data.extend([GeoMax, HarMax, CtyMax, ProMax, ValGeo, GeoLim, ValHar, HarLim, GeoRank, HarRank])

        # Multi_invest_1
        ValGeoNgn, GeoNgn, ValHarNgn, HarNgn = funcs.singleYearThreshold(
            weight, vis.INDEX, vis.PROFIT, vis.INTEREST
        )

        list_invest, list_profit = funcs.singleYearThreshold_test(
            weight1, vis1.INDEX, vis1.PROFIT, vis1.INTEREST, ValHarNgn
        )

        ProNgn1 = list_profit[0]
        CtyNgn1 = "_".join([vis1.data.loc[k, "SYMBOL"] for k in list_invest[0]])

        _, list_profit_test = funcs.singleYearThreshold_test(
            weight, vis.INDEX, vis.PROFIT, vis.INTEREST, ValHarNgn
        )
        list_profit_test = list_profit_test[:-1]
        list_profit_rank = []
        for i in range(len(list_profit_test)):
            list_profit_rank.append(find_rank(vis.sorted_PROFIT[i], list_profit_test[i]))
        list_profit_rank = np.array(list_profit_rank, float)
        RankGeo1 = funcs.geomean(list_profit_rank)
        RankHar1 = funcs.harmean(list_profit_rank)

        temp_data.extend([ValGeoNgn, GeoNgn, ValHarNgn, HarNgn, CtyNgn1, ProNgn1, RankGeo1, RankHar1])

        # Multi_invest_2
        ValGeoNgn2, GeoNgn2, ValHarNgn2, HarNgn2, last_reason = funcs.doubleYearThreshold(
            weight, vis.INDEX, vis.PROFIT, vis.SYMBOL, vis.INTEREST, vis.BOOL_ARG
        )

        list_invest, list_profit = funcs.doubleYearThreshold_test(
            weight2, vis2.INDEX, vis2.PROFIT, vis2.SYMBOL, vis2.INTEREST, vis2.BOOL_ARG, ValHarNgn2, last_reason
        )

        ProNgn2 = list_profit[0]
        CtyNgn2 = "_".join([vis2.data.loc[k, "SYMBOL"] for k in list_invest[0]])

        _, list_profit_test = funcs.doubleYearThreshold_test(
            weight, vis.INDEX, vis.PROFIT, vis.SYMBOL, vis.INTEREST, vis.BOOL_ARG, ValHarNgn2, 0
        )
        list_profit_test = list_profit_test[:-1]
        list_profit_rank = []
        for i in range(len(list_profit_test)):
            list_profit_rank.append(find_rank(vis.sorted_PROFIT[i+1], list_profit_test[i]))
        list_profit_rank = np.array(list_profit_rank, float)
        RankGeo2 = funcs.geomean(list_profit_rank)
        RankHar2 = funcs.harmean(list_profit_rank)

        temp_data.extend([ValGeoNgn2, GeoNgn2, ValHarNgn2, HarNgn2, CtyNgn2, ProNgn2, RankGeo2, RankHar2])

        # Multi_invest_3
        ValGeoNgn3, GeoNgn3, ValHarNgn3, HarNgn3, last_reason = funcs.tripleYearThreshold(
            weight, vis.INDEX, vis.PROFIT, vis.SYMBOL, vis.INTEREST, vis.BOOL_ARG
        )

        list_invest, list_profit = funcs.tripleYearThreshold_test(
            weight3, vis3.INDEX, vis3.PROFIT, vis3.SYMBOL, vis3.INTEREST, vis3.BOOL_ARG, ValHarNgn3, last_reason
        )

        ProNgn3 = list_profit[0]
        CtyNgn3 = "_".join([vis3.data.loc[k, "SYMBOL"] for k in list_invest[0]])

        _, list_profit_test = funcs.tripleYearThreshold_test(
            weight, vis.INDEX, vis.PROFIT, vis.SYMBOL, vis.INTEREST, vis.BOOL_ARG, ValHarNgn3, 0
        )
        list_profit_test = list_profit_test[:-1]
        list_profit_rank = []
        for i in range(len(list_profit_test)):
            list_profit_rank.append(find_rank(vis.sorted_PROFIT[i+2], list_profit_test[i]))
        list_profit_rank = np.array(list_profit_rank, float)
        RankGeo3 = funcs.geomean(list_profit_rank)
        RankHar3 = funcs.harmean(list_profit_rank)

        temp_data.extend([ValGeoNgn3, GeoNgn3, ValHarNgn3, HarNgn3, CtyNgn3, ProNgn3, RankGeo3, RankHar3])

        # Slope
        temp_data.extend(list(funcs.find_slope(
            weight, vis.INDEX, vis.PROFIT, vis.INTEREST
        )))

        #
        all_data.append(temp_data)


    list_column = ["id", "CT"]\
                + "GeoMax, HarMax, CtyMax, ProMax, ValGeo, GeoLim, ValHar, HarLim, GeoRank, HarRank".split(", ")\
                + "ValGeo1, GeoNgn1, ValHar1, HarNgn1, CtyNgn1, ProNgn1, RankGeo1, RankHar1".split(", ")\
                + "ValGeo2, GeoNgn2, ValHar2, HarNgn2, CtyNgn2, ProNgn2, RankGeo2, RankHar2".split(", ")\
                + "ValGeo3, GeoNgn3, ValHar3, HarNgn3, CtyNgn3, ProNgn3, RankGeo3, RankHar3".split(", ")\
                + "Slope_avg, Slope_wgt_avg".split(", ")

    df_new = pd.DataFrame(all_data, columns=list_column)
    return df_new
