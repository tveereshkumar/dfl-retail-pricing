# utils/vendor.py

def compute_vendor_allowance(demand, prices, promos, thresholds):
    allowance = 0

    for i in range(len(demand)):
        if promos[i] == 1 and demand[i] >= thresholds[i]:
            allowance += 2.0 * demand[i]  # example funding
    return allowance