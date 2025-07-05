"""
Copyright (C) 2025 Interactive Brokers LLC. All rights reserved. This code is subject to the terms
 and conditions of the IB API Non-Commercial License or the IB API Commercial License, as applicable.
"""
from decimal import Decimal

from ibapi.contract import Contract
from ibapi.execution import Execution
from ibapi.utils import floatMaxString

def decodeContract(contractProto):
    contract = Contract()
    if (contractProto.HasField('conId')):
        contract.conId = contractProto.conId
    if (contractProto.HasField('symbol')):
        contract.symbol = contractProto.symbol
    if (contractProto.HasField('secType')):
        contract.secType = contractProto.secType
    if (contractProto.HasField('lastTradeDateOrContractMonth')):
        contract.lastTradeDateOrContractMonth = contractProto.lastTradeDateOrContractMonth
    if (contractProto.HasField('strike')):
        contract.strike = contractProto.strike
    if (contractProto.HasField('right')):
        contract.right = contractProto.right
    if (contractProto.HasField('multiplier')):
        contract.multiplier = floatMaxString(contractProto.multiplier)
    if (contractProto.HasField('exchange')):
        contract.exchange = contractProto.exchange
    if (contractProto.HasField('currency')):
        contract.currency = contractProto.currency
    if (contractProto.HasField('localSymbol')):
        contract.localSymbol = contractProto.localSymbol
    if (contractProto.HasField('tradingClass')):
        contract.tradingClass = contractProto.tradingClass
    return contract

def decodeExecution(executionProto):
    execution = Execution()
    if (executionProto.HasField('orderId')):
        execution.orderId = executionProto.orderId
    if (executionProto.HasField('clientId')):
        execution.clientId = executionProto.clientId
    if (executionProto.HasField('execId')):
        execution.execId = executionProto.execId
    if (executionProto.HasField('time')):
        execution.time = executionProto.time
    if (executionProto.HasField('acctNumber')):
        execution.acctNumber = executionProto.acctNumber
    if (executionProto.HasField('exchange')):
        execution.exchange = executionProto.exchange
    if (executionProto.HasField('side')):
        execution.side = executionProto.side
    if (executionProto.HasField('shares')):
        execution.shares = Decimal(executionProto.shares)
    if (executionProto.HasField('price')):
        execution.price = executionProto.price
    if (executionProto.HasField('permId')):
        execution.permId = executionProto.permId
    if (executionProto.HasField('isLiquidation')):
        execution.liquidation = 1 if executionProto.isLiquidation else 0
    if (executionProto.HasField('cumQty')):
        execution.cumQty = Decimal(executionProto.cumQty)
    if (executionProto.HasField('avgPrice')):
        execution.avgPrice = executionProto.avgPrice
    if (executionProto.HasField('orderRef')):
        execution.orderRef = executionProto.orderRef
    if (executionProto.HasField('evRule')):
        execution.evRule = executionProto.evRule
    if (executionProto.HasField('evMultiplier')):
        execution.evMultiplier = executionProto.evMultiplier
    if (executionProto.HasField('modelCode')):
        execution.modelCode = executionProto.modelCode
    if (executionProto.HasField('lastLiquidity')):
        execution.lastLiquidity = executionProto.lastLiquidity
    if (executionProto.HasField('isPriceRevisionPending')):
        execution.pendingPriceRevision = executionProto.isPriceRevisionPending
    if (executionProto.HasField('submitter')):
        execution.submitter = executionProto.submitter
    return execution

