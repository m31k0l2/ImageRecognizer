val image = MNIST.buildBatch(10).first()

fun main(args: Array<String>) {
    changeStructure("nets/nw.net", "nets/nw.net")
}

fun changeStructure(from: String, to: String) {
    val nw = NetworkIO().load(from)!!
    val net = ImageNetEvolution().createNet()
    changeNeurons(0, net, nw)
    net.activate(image)
    NetworkIO().save(net, to)
}

private fun changeNeurons(layerNumber: Int, net: CNetwork, nw: Network) {
    net.layers[layerNumber].neurons.clear()
    net.layers[layerNumber].neurons.addAll(nw.layers[layerNumber].neurons)
}
