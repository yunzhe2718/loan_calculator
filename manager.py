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



# # In[2]:


# # raw_input = pd.read_excel('Housing loan.xlsx', sheet_name='cash flow')
# xls = pd.ExcelFile('Housing loan copy.xlsx')
# income_sheet = pd.read_excel(xls, 'income')
# rates_sheet = pd.read_excel(xls, 'rates')


# # In[4]:


# income_data, financial_data = trim(income_data, financial_data)

# max_days = (financial_data['last date'] - financial_data['first date']).days + 1

# # for k, v in financial_data.items():
# #     print(k, v, sep='\t')


# input_visualise(income_data, financial_data)

# ops_cum, actual_transaction_days, ops, intr_earned = optimiser(income_data, financial_data)

# print(actual_transaction_days[-1], intr_earned)

# optimised_day = len(ops_cum[0])
# ops_cum = np.pad(ops_cum, ((0,0), (0, max_days - optimised_day)), mode='edge')
# income_cum = np.cumsum(income_data, axis=2)
# new_income = ([1] + financial_data['exchange rates']) @ (income_cum[0] + ops_cum)
# old_income = ([1] + financial_data['exchange rates']) @ income_cum[0]
# old_expenses = ([1] + financial_data['exchange rates']) @ income_cum[1]
# plt.plot(old_income, label='in')
# plt.plot(old_expenses, label='out')
# plt.plot(new_income, label='in - deposits')
# plt.legend()
# plt.show()

# headers = [[financial_data['names'][i] + ' ' + str(int(op[0])) + 'M' for op in cur] for i, cur in enumerate(financial_data['deposit options'])]
# headers.append([cur + ' ex' for cur in financial_data['names'][1:]])
# headers = np.concatenate(headers)
# transaction_data = [(d, op) for d, op in zip(headers,ops) if np.max(op) > 0.01]

# transaction_plan = pd.DataFrame(data=np.transpose(np.array([d[1] for d in transaction_data])), columns=[d[0] for d in transaction_data], index=actual_transaction_days)
# transaction_plan.replace(0, '')


# # In[3]:


# total_price = 850000
# down_payment = 120000 
# duration = 5
# annual_intr_rate = 0.03 
# first_instalment_date = datetime.date(2025, 1, 1)
# starting_cash = 20000
# transaction_interval = relativedelta(months=1)


# last_instalment_date = first_instalment_date + relativedelta(years=duration) + relativedelta(months=3)
# max_days = (last_instalment_date - first_instalment_date).days + 1

# # decide the list of transaction days
# transaction_days = []
# current_ind = 0
# while current_ind < max_days:
#     transaction_days.append(current_ind)
#     next_day = first_instalment_date + relativedelta(days=current_ind) + transaction_interval
#     current_ind = (next_day - first_instalment_date).days


# # extract financial informations
# intr = rates_sheet[['1 M', '3 M', '6 M', '9 M', '12 M']]
# tenures = [1,3,6,9,12]
# depo_options = []
# for i, val in intr.iterrows():
#     opts = [(tenure, r) for tenure, r in zip(tenures, val) if not np.isnan(r)]
#     depo_options.append(opts)

# ex_rates = list(np.array(rates_sheet['exchange rate'][1:]))
# names = list(rates_sheet['currency'])

# # input dictionary
# financial_data = {
#     'names': names,
#     'deposit options': depo_options,
#     'buffer time': relativedelta(days=0),
#     'exchange rates': ex_rates,
#     'exchange penalty': 0.01,
#     'first date': first_instalment_date,
#     'last date': last_instalment_date,
#     'transaction days': transaction_days
# }

# income_dates = np.array(income_sheet['maturation'].dt.date)
# income = [index_days(list(zip(income_dates, np.array(income_sheet[c]))), 
#                      financial_data['first date'], financial_data['last date']) for c in names]
# income[0][0] += starting_cash

# future_payment_list = instalment(total_price, down_payment, duration, annual_intr_rate, first_instalment_date)
# expenses = [index_days(future_payment_list, financial_data['first date'], financial_data['last date'])]
# expenses = np.pad(expenses, ((0, len(names) - 1),(0,0)))
# income_data = np.round(np.array([income, expenses]), 2)


# In[ ]:


# def ltr_matrix_sparse(n):
#     return sp.sparse.csr_array(np.tril(np.ones(n)))

# def ltr_matrix(n):
#     return np.tril(np.ones(n))

# def slice_flow(b, i):
#     return np.reshape(b, (3, -1)).transpose()[:i].transpose().flatten()

# # the ith column vector is given by [0,0,..., 0, -1, 0, ... 0, 1 + int_rate, 0,...,0]. 
# def depo_matrix_sparse(first_date, n_days, tenure_in_months, normalised_rate):
#     depo_entries = sp.sparse.identity(n_days)
#     coord_2 = list(range(n_days))
#     coord_1 = []
#     # save on the ith day after first_date, calculate the maturation day is on which day after the first day.
#     # e.g. if the first day is 1 - Jan, save on 4 - Jan, tenure_in_months = 1, then the maturation is on 4 - Feb. 
#     # The coordinate should then be 31 + 3 = 34
#     for i in range(n_days):
#         maturation_date = first_date + relativedelta(days=i) + relativedelta(months=tenure_in_months)
#         difference = (maturation_date - first_date).days
#         coord_1.append(difference)
    
#     coord = (coord_1, coord_2)
#     mature_entries = sp.sparse.csr_array(([1 + normalised_rate] * n_days, coord))
#     mature_entries = mature_entries[:n_days] # make it into a square matrix
#     m = mature_entries - depo_entries
#     m = np.tril(np.ones(n_days)) @ m
#     return sp.sparse.csr_matrix(m)

# def depo_matrix(first_date, n_days, tenure_in_months, normalised_rate, sparse=False):
#     if sparse:
#         depo_entries = sp.sparse.identity(n_days)
#         coord_2 = list(range(n_days))
#         coord_1 = []
#         # save on the ith day after first_date, calculate the maturation day is on which day after the first day.
#         # e.g. if the first day is 1 - Jan, save on 4 - Jan, tenure_in_months = 1, then the maturation is on 4 - Feb. 
#         # The coordinate should then be 31 + 3 = 34
#         for i in range(n_days):
#             maturation_date = first_date + relativedelta(days=i) + relativedelta(months=tenure_in_months)
#             difference = (maturation_date - first_date).days
#             coord_1.append(difference)
        
#         coord = (coord_1, coord_2)
#         mature_entries = sp.sparse.csr_array(([1 + normalised_rate] * n_days, coord))
#         mature_entries = mature_entries[:n_days] # make it into a square matrix
#         m = mature_entries - depo_entries
#         m = np.tril(np.ones(n_days)) @ m
#         result = sp.sparse.csr_matrix(m)
#     else:
#         mat = []
#         # the ith column vector is given by [0,0,..., 0, -1, -1, ... -1, 1 + int_rate, 0,...,0].
#         # there are i starting zeros, and as many -1 as days during the tenure.
#         for i in range(n_days):
#             row = [0] * i
#             maturation_date = first_date + relativedelta(days=i) + relativedelta(months=tenure_in_months)
#             tenure_in_days = (maturation_date - first_date).days - i
#             if i + tenure_in_days <= n_days - 1:
#                 row += [-1] * tenure_in_days + [normalised_rate] * (n_days - i - tenure_in_days)
#             else:
#                 row += [-1] * (n_days - i)
#             mat.append(row)
#         result = np.transpose(np.array(mat))
#     return result



# In[ ]:


# def contr_data_sparse(depo_matrices, ex_rates, exchange_penalty, target_day):
#     depo_matrices_stacked = sp.sparse.block_diag([sp.sparse.hstack([m[:target_day, :target_day] for m in l]) for l in depo_matrices])

#     ex_matrix = sp.sparse.block_array([[sp.sparse.hstack([r * (1 - exchange_penalty) * ltr_matrix_sparse(target_day) for r in ex_rates])],
#                                        [sp.sparse.block_diag([-ltr_matrix_sparse(target_day)] * len(ex_rates))]])

#     # We need A x + b >= 0 and maximise obj
#     A = sp.sparse.hstack([depo_matrices_stacked, ex_matrix])
#     A2 = sp.sparse.csr_matrix(A)
#     obj = A2[target_day -1]
#     for j in range(len(ex_rates)):
#         obj += A2[(j + 1) * target_day + target_day -1] * ex_rates[j]
#     return (A, obj.toarray()[0])

# def contr_data(depo_matrices, ex_rates, exchange_penalty, target_day, sparse=False):
#     if sparse:
#         depo_matrices_stacked = sp.sparse.block_diag([sp.sparse.hstack([m[:target_day, :target_day] for m in l]) for l in depo_matrices])
    
#         ex_matrix = sp.sparse.block_array([[sp.sparse.hstack([r * (1 - exchange_penalty) * ltr_matrix_sparse(target_day) for r in ex_rates])],
#                                            [sp.sparse.block_diag([-ltr_matrix_sparse(target_day)] * len(ex_rates))]])
    
#         # We need A x + b >= 0 and maximise obj
#         A = sp.sparse.hstack([depo_matrices_stacked, ex_matrix])
#         A2 = sp.sparse.csr_matrix(A)
#         obj = A2[target_day -1]
#         for j in range(len(ex_rates)):
#             obj += A2[(j + 1) * target_day + target_day -1] * ex_rates[j]
#         result = (A, obj.toarray()[0])
#     else:
#         depo_matrices_stacked = sp.linalg.block_diag(*[np.hstack([m[:target_day, :target_day] for m in l]) for l in depo_matrices])
    
#         ex_matrix = np.block([[np.hstack([r * (1 - exchange_penalty) * ltr_matrix(target_day) for r in ex_rates])],
#                                            [sp.linalg.block_diag(*[-ltr_matrix(target_day)] * len(ex_rates))]])
    
#         # We need A x + b >= 0 and maximise obj
#         A = np.hstack([depo_matrices_stacked, ex_matrix])
#         obj = A[target_day -1].copy()
#         # print(obj)
#         for j in range(len(ex_rates)):
#             # print(A[(j + 1) * target_day + target_day -1])
#             obj += A[(j + 1) * target_day + target_day -1] * ex_rates[j]
#         result = (A, obj)
#     return result
    


# In[ ]:





# In[ ]:





# In[ ]:




