def allocate_warehouse(order_location, warehouses):

    if not warehouses:
        return {
            "selected_warehouse": None,
            "distance": None,
            "message": "No warehouses available"
        }

    best_warehouse = None
    best_distance = float("inf")

    order_lat = order_location["latitude"]
    order_lon = order_location["longitude"]

    for w in warehouses:

        w_lat = w["latitude"]
        w_lon = w["longitude"]

        distance = ((order_lat - w_lat) ** 2 +
                    (order_lon - w_lon) ** 2) ** 0.5

        if distance < best_distance:
            best_distance = distance
            best_warehouse = w

    return {
        "selected_warehouse": best_warehouse,
        "distance": best_distance
    }