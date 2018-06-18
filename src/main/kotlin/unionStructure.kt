
fun main(args: Array<String>) {
    union(7, "nets/789", 3)
}

fun union(size: Int, dir: String, outputSize: Int) {
    val nets = (1..size).map { "$dir/nw$it.net" }.mapNotNull { CNetwork().load(it) }
    val structure = (1..size).flatMap { List(5, { 0 }) }.toIntArray()
    val nw = buildMultiNetwork(size, *structure, 10, outputSize)
    for (i in 0 until size) {
        (0..4).forEach { nw.layers[it+5*i].neurons.addAll(nets[i].layers[it].neurons) }
    }
    nw.activate(MNIST.buildBatch(10).first(), 1.0)
    nw.save("nets/nw.net")
}

fun unionTotal(size: Int, dir: String, outputSize: Int) {
    val nets = (1..size).map { "$dir/nw$it.net" }.mapNotNull { CNetwork().load(it) }
    val structure = (1..size).flatMap { List(5, { 0 }) }.toIntArray()
    val nw = buildMultiNetwork(size, *structure, 10, outputSize)
    for (i in 0 until size) {
        (0..4).forEach { nw.layers[it+5*i].neurons.addAll(nets[i].layers[it].neurons) }
    }
    nw.activate(MNIST.buildBatch(10).first(), 1.0)
    nw.save("nets/nw.net")
}