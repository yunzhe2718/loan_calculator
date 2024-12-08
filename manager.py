#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy as sp
import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import matplotlib.pyplot as plt

def depo_matrix(first_date, n_days, transaction_days, tenure_in_months, buffer_time, annual_rate):
    mat = []
    for i in transaction_days:
        deposit_date = first_date + relativedelta(days=i) 
        maturation_date = deposit_date + relativedelta(months=tenure_in_months)
        buffered_maturation_date = maturation_date + buffer_time
        tenure_in_days = (maturation_date - deposit_date).days
        buffered_waiting_days = (buffered_maturation_date - deposit_date).days
        if i + buffered_waiting_days <= n_days - 1:
            row = [0] * i + [-1] * buffered_waiting_days + [annual_rate * tenure_in_days / 360] * (n_days - i - buffered_waiting_days)
            mat.append(row)
        else:
            break
    result = np.transpose(np.array(mat))
    return result

def ltr(first_date, n_days, transaction_days):
    mat = []
    for i in transaction_days:
        if i < n_days:
            row = [0] * i + [1] * (n_days - i)
            mat.append(row)
        else:
            break
    result = np.transpose(np.array(mat))
    return result

def assemble(depo_matrices, ltr_mat, ex_rates, ex_penalty):
    depo = sp.linalg.block_diag(*[np.hstack(l) for l in depo_matrices])
    exchange = np.block([[np.hstack([r * (1 - ex_penalty) * ltr_mat for r in ex_rates])],
                         [sp.linalg.block_diag(*[-ltr_mat for r in ex_rates])]])
    A = np.hstack([depo, exchange])
    return A

def last_day(A, ex_rates):
    n_days = int(A.shape[0] / (1 + len(ex_rates)))
    row = A[n_days -1].copy()
    for (i, r) in enumerate(ex_rates):
        row += A[(i + 1) * n_days + n_days -1] * r                        
    return row

def transaction_matrix(financial_data, n_days):
    transaction_days = financial_data['transaction days']

    n_currency = len(financial_data['names'])
    depo_matrices = [] 
    lengths = []
    for opts in financial_data['deposit options']:
        l = []
        for tenure, rate in opts:
            m = depo_matrix(financial_data['first date'], n_days, transaction_days, 
                            tenure, financial_data['buffer time'], rate)
            # print(m)
            if len(m) > 0:
                l.append(m)
                lengths.append(len(m[0]))
            else:
                continue
        depo_matrices.append(l)

    # print(financial_data['deposit options'])
    ltr_mat = ltr(financial_data['first date'], n_days, transaction_days)
    lengths += [len(ltr_mat[0])] * (n_currency - 1)
    A = assemble(depo_matrices, ltr_mat, financial_data['exchange rates'], financial_data['exchange penalty'])
    obj = last_day(A, financial_data['exchange rates'])
    return (A, obj, lengths)

def cash_flow_raw(income_data, financial_data, n_days):
    data = income_data[:, :, :n_days]
    
    income_cum = np.cumsum(data[0], axis=1)
    expenses_cum = np.cumsum(data[1], axis=1)
    raw_balance = income_cum - expenses_cum
    # print(raw_balance.shape)
    b = np.concatenate(raw_balance)
    combi = np.round(([1] + financial_data['exchange rates']) @ raw_balance, 2)
    return (b, combi)

def trim(income_data, financial_data):
    n_currency = min(1 + len(financial_data['exchange rates']), len(financial_data['deposit options']), 
                     len(financial_data['names']), len(income_data[0]), len(income_data[1]))
    financial_data['exchange rates'] = financial_data['exchange rates'][:n_currency - 1]
    financial_data['deposit options'] = financial_data['deposit options'][:n_currency]
    financial_data['names'] = financial_data['names'][:n_currency]
    income_data = income_data[:, :n_currency, ...]
    # financial_data['n_currency'] = n_currency
    max_days = (financial_data['last date'] - financial_data['first date']).days + 1
    income_data = np.pad(income_data, ((0,0),(0,0),(0, max(0, max_days - len(income_data[0,0])))))[:, :, :max_days]
        
    return (income_data, financial_data)

def input_visualise(income_data, financial_data):
    names = financial_data['names']
    n_currency = len(financial_data['names'])
    
    fig, axs = plt.subplots(n_currency, figsize=(8, 1.8 * n_currency)) 
    
    for i in range(n_currency):
        axs[i].plot(np.cumsum(income_data[0, i]), label=names[i] + ' in')
        axs[i].plot(np.cumsum(income_data[1, i]), label=names[i] + ' out')
        axs[i].legend()

    plt.show()
    return

def optimiser(income_data, financial_data):
    max_days = (financial_data['last date'] - financial_data['first date']).days + 1
    transaction_days = financial_data['transaction days']

    # begin LP procedures
    optimised_day = 0
    
    _, combi = cash_flow_raw(income_data, financial_data, max_days)
    if combi[-1] > 0:
        target_day = max_days
    else:
        target_day = np.argmax(combi < 0)
    
    while optimised_day < max_days:
        if target_day == 0:
            print('Error')
            break
    
        A, obj, lengths = transaction_matrix(financial_data, target_day)
        b, _ = cash_flow_raw(income_data, financial_data, target_day)
        # plt.imshow(A)
        # plt.colorbar()
        # plt.show()
        print('running LP, targeting day', target_day)
        # print(*combi[target_day - 3:target_day + 3])
        result = sp.optimize.linprog(-obj, A_ub=-A, b_ub=b, method='highs')
        # print(result)
        intr_earned = -result.fun
        print(combi[target_day-1], combi[target_day-1] + intr_earned)
        
        optimised_day = target_day
        if combi[-1] + intr_earned > 0:
            target_day = max_days
        elif np.argmax(combi + intr_earned < 0) > target_day:
            target_day = np.argmax(combi + intr_earned < 0)
        else:
            break
    # print('total interests:', intr_earned)
    
    # process output
    ops_cum = np.reshape(A @ result.x, (-1, optimised_day))
    actual_transaction_days = [financial_data['first date'] + relativedelta(days=i) for i in transaction_days[:lengths[-1]]]
    split_points = np.cumsum(lengths)[:-1]

    ops = np.round(result.x, decimals=2)
    ops_split = np.hsplit(ops, split_points)
    ops = [np.pad(l, (0, lengths[-1] - len(l))) for l in ops_split]

    return (ops_cum, actual_transaction_days, ops, intr_earned)


def instalment(total_price, down_payment, duration, annual_intr_rate, first_instalment_date, aggressive=True):
    p = total_price - down_payment
    n = duration * 12
    r = annual_intr_rate / 12
    date_list = [first_instalment_date + relativedelta(months=i) for i in range(n)]
    if aggressive:
        instalments = [p / n +(p - p * k / n) * r for k in range(n)]
    else:
        instalments = [(p * r * (1 + r) ** n) / ((1+r)**n - 1)] * n
    result = list(zip(date_list, instalments))
    return result

def index_days(list_of_pairs, start_day, finish_day):
    list_of_pairs = [p for p in list_of_pairs if p[0] >= start_day and p[0] <= finish_day]
    result = []
    for d, val in list_of_pairs:
        ind = (d - start_day).days
        result += [0] * (ind - len(result)) + [val]
    result += [0] * (finish_day - list_of_pairs[-1][0]).days
    return np.array(result)

