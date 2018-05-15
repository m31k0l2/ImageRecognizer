data class Shop(val name: String, val customers: List<Customer>)

data class Customer(val name: String, val city: City, val orders: List<Order>)

data class Order(val products: List<Product>, val isDelivered: Boolean)

data class Product(val name: String, val price: Double)

data class City(val name: String)

// Return the set of products that were ordered by every customer
fun Shop.getSetOfProductsOrderedByEveryCustomer(): Set<Product> {
    val products = customers.map { it.orders.flatMap { it.products }.toSet() }
    return products.fold(products.first(), { everyCustomerProducts, customerProducts ->
        everyCustomerProducts.filter { it in customerProducts }.toSet()
    })
}

