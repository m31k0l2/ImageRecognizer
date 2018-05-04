val image = MNIST.buildBatch(10).first()

fun main(args: Array<String>) {
    changeStructure("nets/nw.net", "nets/nw.net")
}

fun changeStructure(from: String, to: String) {
    val nw = NetworkIO().load(from)!!
    val net = ImageNetEvolution().createNet()
    changeNeurons(0, 3, net, nw)
    net.activate(image)
    NetworkIO().save(net, to)
}

private fun changeNeurons(savedLayerNumberFrom: Int, savedLayerNumberTo: Int, net: CNetwork, nw: Network) {
    for (i in savedLayerNumberFrom..savedLayerNumberTo) {
        changeNeurons(i, net, nw)
    }
}

private fun changeNeurons(savedLayerNumber: Int, net: CNetwork, nw: Network) {
    net.layers[savedLayerNumber].neurons.clear()
    net.layers[savedLayerNumber].neurons.addAll(nw.layers[savedLayerNumber].neurons)
}
