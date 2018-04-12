val image = MNIST.buildBatch(10).first()

fun main(args: Array<String>) {
    changeStructure("nets/nw.net", "nets/nw.net")
}

fun changeStructure(from: String, to: String) {
    val nw = NetworkIO().load(from)!!
    val net = CNetwork(0, 0, 4, 0, 0, 0)
    net.layers[0].neurons.addAll(nw.layers[0].neurons)
    net.layers[1].neurons.addAll(nw.layers[1].neurons)
//    net.layers[2].neurons.addAll(nw.layers[2].neurons)
    net.layers[3].neurons.addAll(nw.layers[3].neurons)
    net.layers[4].neurons.addAll(nw.layers[4].neurons)
    net.layers[5].neurons.addAll(nw.layers[5].neurons)
    net.activate(image)
    NetworkIO().save(net, to)
}
