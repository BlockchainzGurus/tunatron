from app.service import list_orders

def orders_handler(request):
    uid = request.args["user_id"]
    return list_orders(uid)
