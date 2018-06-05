val nw1 = CNetwork().load("nets/nw01234567_1-3-2-3-60-10.net")!!
val nw2 = CNetwork().load("nets/nw6789_3-2-2-1-40-10.net")!!
fun main(args: Array<String>) {
    val nw = buildNetwork(0, 0, 0, 0, 80, 10)
    for (i in 0..3) {
        if (nw.layers[i].neurons.size > 0) continue
        nw.layers[i].neurons.addAll(nw1.layers[i].neurons)
        if (i < 0) continue
        nw.layers[i].neurons.addAll(nw2.layers[i].neurons)
    }
    nw.activate(MNIST.buildBatch(10).first(), 1.0)
    nw.save("nets/nw.net")
}