val nw1 = NetworkIO().load("nets/nw012.net")!!
val nw2 = NetworkIO().load("nets/nw1467.net")!!
fun main(args: Array<String>) {
    val nw = CNetwork(0, 0, 0, 0, 10, 10)
    for (i in 0..3) {
        nw.layers[i].neurons.addAll(nw1.layers[i].neurons)
        nw.layers[i].neurons.addAll(nw2.layers[i].neurons)
    }
    nw.activate(image)
    NetworkIO().save(nw, "nets/nw.net")
}