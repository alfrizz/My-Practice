"""
Copyright (C) 2025 Interactive Brokers LLC. All rights reserved. This code is subject to the terms
 and conditions of the IB API Non-Commercial License or the IB API Commercial License, as applicable.
"""


from ibapi.object_implem import Object
from ibapi.const import UNSET_DECIMAL
from ibapi.const import UNSET_INTEGER
from ibapi.utils import decimalMaxString
from ibapi.utils import intMaxString
from ibapi.utils import floatMaxString
from ibapi.utils import longMaxString


class Execution(Object):
    def __init__(self):
        self.execId = ""
        self.time = ""
        self.acctNumber = ""
        self.exchange = ""
        self.side = ""
        self.shares = UNSET_DECIMAL
        self.price = 0.0
        self.permId = 0
        self.clientId = 0
        self.orderId = 0
        self.liquidation = 0
        self.cumQty = UNSET_DECIMAL
        self.avgPrice = 0.0
        self.orderRef = ""
        self.evRule = ""
        self.evMultiplier = 0.0
        self.modelCode = ""
        self.lastLiquidity = 0
        self.pendingPriceRevision = False
        self.submitter = ""

    def __str__(self):
        return (
            "ExecId: %s, Time: %s, Account: %s, Exchange: %s, Side: %s, Shares: %s, Price: %s, PermId: %s, "
            "ClientId: %s, OrderId: %s, Liquidation: %s, CumQty: %s, AvgPrice: %s, OrderRef: %s, EvRule: %s, "
            "EvMultiplier: %s, ModelCode: %s, LastLiquidity: %s, PendingPriceRevision: %s, Submitter: %s"
            % (
                self.execId,
                self.time,
                self.acctNumber,
                self.exchange,
                self.side,
                decimalMaxString(self.shares),
                floatMaxString(self.price),
                longMaxString(self.permId),
                intMaxString(self.clientId),
                intMaxString(self.orderId),
                intMaxString(self.liquidation),
                decimalMaxString(self.cumQty),
                floatMaxString(self.avgPrice),
                self.orderRef,
                self.evRule,
                floatMaxString(self.evMultiplier),
                self.modelCode,
                intMaxString(self.lastLiquidity),
                self.pendingPriceRevision,
                self.submitter,
            )
        )


class ExecutionFilter(Object):
    # Filter fields
    def __init__(self):
        self.clientId = 0
        self.acctCode = ""
        self.time = ""
        self.symbol = ""
        self.secType = ""
        self.exchange = ""
        self.side = ""
        self.lastNDays = UNSET_INTEGER
        self.specificDates = None
