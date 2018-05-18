fun main(args: Array<String>) {
    changeStructure("nets/nw.net", "nets/nw.net", listOf(0,1,2,3))
}

fun changeStructure(from: String, to: String, layers: List<Int>) {
    val nw = NetworkIO().load(from)!!
    val net = ImageNetEvolution().createNet()
    layers.forEach { changeNeurons(it, net, nw) }
    net.activate(MNIST.buildBatch(10).first())
    NetworkIO().save(net, to)
}

private fun changeNeurons(savedLayerNumber: Int, net: CNetwork, nw: Network) {
    val neurons = net.layers[savedLayerNumber].neurons
    val source = nw.layers[savedLayerNumber].neurons
    for (it in 0 until source.size) {
        neurons[it] = source[it]
    }
}
