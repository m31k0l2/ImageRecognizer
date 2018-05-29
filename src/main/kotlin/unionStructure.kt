val nw1 = NetworkIO().load("nets/nw012_2212.net")!!
val nw2 = NetworkIO().load("nets/nw345_2311.net")!!
fun main(args: Array<String>) {
    val nw = CNetwork(0, 4, 4, 4, 40, 10)
    for (i in 0..3) {
        if (nw.layers[i].neurons.size > 0) continue
        nw.layers[i].neurons.addAll(nw1.layers[i].neurons)
        nw.layers[i].neurons.addAll(nw2.layers[i].neurons)
    }
    nw.activate(MNIST.buildBatch(10).first())
    NetworkIO().save(nw, "nets/nw.net")
}