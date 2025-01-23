def recommend_product(key_points):
    products = {
        "perfume": {"name": "Luxury Perfume", "price": 140.0, "category": "Beauty"},
        "makeup": {"name": "Makeup Kit", "price": 220.0, "category": "Beauty"},
        "lotion": {"name": "Body Lotion", "price": 66.5, "category": "Beauty"},
        "hair": {"name": "Hair Straightener", "price": 400.0, "category": "Beauty"}
    }

    recommendations = [products[key.lower()] for key in key_points if key.lower() in products]

    if recommendations:
        return recommendations
    else:
        # Provide a fallback recommendation instead of "No specific product found"
        return [{"name": "Popular Beauty Product", "price": 99.99, "category": "Beauty"}]
