def allocate_warehouse(order_location, warehouses):

    best_warehouse = None
    best_distance = float("inf")

    for w in warehouses:

        wx = w["x"]
        wy = w["y"]

        distance = ((order_location["x"] - wx) ** 2 +
                    (order_location["y"] - wy) ** 2) ** 0.5

        if distance < best_distance:
            best_distance = distance
            best_warehouse = w["name"]

    return {
        "selected_warehouse": best_warehouse,
        "distance": best_distance
    }