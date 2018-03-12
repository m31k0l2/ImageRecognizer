fun main(args: Array<String>) {
    val batch = MNIST.buildBatch(10)
    val nw02 = NetworkIO().load("nets/nw02.net")!!
    val nw13 = NetworkIO().load("nets/nw13.net")!!
    val nw24 = NetworkIO().load("nets/nw24.net")!!
    val nw35 = NetworkIO().load("nets/nw35.net")!!
    val nw46 = NetworkIO().load("nets/nw46.net")!!
    val nw57 = NetworkIO().load("nets/nw57.net")!!
    val nw68 = NetworkIO().load("nets/nw68.net")!!
    val nw79 = NetworkIO().load("nets/nw79.net")!!
    batch.forEach {
        print("${it.index} -> nw02: " + recognize(nw02, it))
        print(", nw13: " + recognize(nw13, it))
        print(", nw24: " + recognize(nw24, it))
        print(", nw35: " + recognize(nw35, it))
        print(", nw46: " + recognize(nw46, it))
        print(", nw57: " + recognize(nw57, it))
        print(", nw68: " + recognize(nw68, it))
        print(", nw79: " + recognize(nw79, it))
        println()
    }
}

private fun recognize(nw: Network, image: Image): String {
    val o = nw.activate(image.colorsMatrix)
    val x = o.max()!!
    val y = o.indexOf(x)
    return if (x > 0.5) {
        "$y[${(x * 1000).toInt() / 10.0}%]"
    } else {
        "?"
    }
}