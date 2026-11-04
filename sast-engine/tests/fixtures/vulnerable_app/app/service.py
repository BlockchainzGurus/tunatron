from app.db import fetch_orders, fetch_user

def list_orders(uid):
    return fetch_orders(uid)

def get_profile(uid):
    return fetch_user(uid)
