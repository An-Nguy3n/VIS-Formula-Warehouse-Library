from pandas import DataFrame, Series, concat
from numpy import array, transpose, zeros, isinf, isnan
from numba import njit


class Base:
    def __init__(self, data: DataFrame, interest: float, valuearg_threshold = 0.0):
        data = data.reset_index(drop=True)
        data.fillna(0.0, inplace=True)

        # Check cac cot bat buoc
        drop_cols = ["TIME", "PROFIT", "SYMBOL", "VALUEARG"]
        for col in drop_cols:
            if col not in data.columns:
                raise Exception(f"Thieu cot {col}")
        
        # Check dtype cua TIME, PROFIT va VALUEARG
        if data["TIME"].dtype != "int64":
            raise Exception("TIME's dtype must be int64")
        if data["PROFIT"].dtype != "float64":
            raise Exception("PROFIT's dtype must be float64")
        if data["VALUEARG"].dtype not in ["int64", "float64"]:
            raise Exception("VALUEARG's dtype must be int64 or float64")
        
        # Check thu tu cot TIME va min PROFIT, min VALUEARG
        if data["TIME"].diff().max() > 0:
            raise Exception("Cot TIME phai giam dan")
        if data["PROFIT"].min() < 0.0:
            raise Exception("PROFIT < 0.0")
        if data["VALUEARG"].min() < 0.0:
            raise Exception("VALUEARG < 0.0")
        
         # INDEX
        index = []
        for i in range(data["TIME"].max(), data["TIME"].min()-1, -1):
            if i not in  data["TIME"].values:
                raise Exception(f"Thieu chu ky {i}")

            index.append(data[data["TIME"]==i].index[0])

        index.append(data.shape[0])
        self.INDEX = array(index)

        # Check SYMBOL co unique trong tung chu ky hay khong
        for i in range(self.INDEX.shape[0] - 1):
            start, end = self.INDEX[i], self.INDEX[i+1]
            if len(data.loc[start:end-1, "SYMBOL"].unique()) != (end - start):
                raise Exception("SYMBOL khong unique o tung chu ky")
        
         # Loai cac cot co kieu du lieu khong phai int64 va float64
        for col in data.columns:
            if col not in drop_cols and data[col].dtype not in ["int64", "float64"]:
                drop_cols.append(col)

        self.drop_cols = drop_cols
        print("Cac cot khong duoc coi la bien:", self.drop_cols)

        # Attrs
        self.data = data
        self.INTEREST = interest
        self.PROFIT = array(data["PROFIT"], float)
        self.PROFIT[self.PROFIT < 5e-324] = 5e-324
        self.VALUEARG = array(data["VALUEARG"], float)
        self.BOOL_ARG = self.VALUEARG >= valuearg_threshold

        symbol_name = data["SYMBOL"].unique()
        self.symbol_name = {symbol_name[i]:i for i in range(len(symbol_name))}
        self.SYMBOL = array([self.symbol_name[s] for s in data["SYMBOL"]])
        self.symbol_name = {v:k for k,v in self.symbol_name.items()}

        operand_data = data.drop(columns=drop_cols)
        operand_name = operand_data.columns
        self.operand_name = {i:operand_name[i] for i in range(len(operand_name))}
        self.OPERAND = transpose(array(operand_data, float))

        self.PROFIT_RANK = array([0.0]*data.shape[0])
        self.PROFIT_RANK_NI = array([0.0]*(self.INDEX.shape[0]-1))
        temp_serie = Series([self.INTEREST])
        for i in range(self.INDEX.shape[0]-1):
            start, end = self.INDEX[i], self.INDEX[i+1]
            temp_ = concat([data.loc[start:end-1, "PROFIT"], temp_serie], ignore_index=True)
            temp_rank = array(temp_.rank(method="min"), float) / (end-start+1)
            self.PROFIT_RANK[start:end] = temp_rank[:-1]
            self.PROFIT_RANK_NI[i] = temp_rank[-1]


@njit
def calculate_formula(formula, operand):
    temp_0 = zeros(operand.shape[1])
    temp_1 = zeros(operand.shape[1])
    temp_op = -1
    num_operand = operand.shape[0]
    for i in range(1, formula.shape[0], 2):
        if formula[i] >= num_operand:
            raise

        if formula[i-1] < 2:
            temp_op = formula[i-1]
            temp_1 = operand[formula[i]].copy()
        else:
            if formula[i-1] == 2:
                temp_1 *= operand[formula[i]]
            else:
                temp_1 /= operand[formula[i]]
        
        if i+1 == formula.shape[0] or formula[i+1] < 2:
            if temp_op == 0:
                temp_0 += temp_1
            else:
                temp_0 -= temp_1
    
    temp_0[isnan(temp_0)] = -1.7976931348623157e+308
    temp_0[isinf(temp_0)] = -1.7976931348623157e+308
    return temp_0
