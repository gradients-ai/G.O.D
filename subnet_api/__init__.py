"""Public, read-only SN56 API (sn56.io).

A thin proxy in front of the validator: it holds the validator API key, rewrites paths, filters
responses through public models so private miner fields never leave, and caches briefly so
public traffic does not load the validator.
"""
