import kotlin.math.min

fun changeStructure(from: String, to: String, layers: List<Int>, filters: Map<Int, List<Int>>, net: Network) {
    val nw = CNetwork().load(from)!!
    layers.forEach {
        val filter = if (it in filters.keys) filters[it]!! else emptyList()
        changeNeurons(it, net, nw, filter)
    }
    net.activate(MNIST.buildBatch(10).first(), 15.0)
    net.save(to)
}

private fun changeNeurons(savedLayerNumber: Int, net: Network, nw: Network, filter: List<Int>) {
    val neurons = net.layers[savedLayerNumber].neurons
    var source = nw.layers[savedLayerNumber].neurons
    if (!filter.isEmpty()) source = source.filterIndexed { index, _ -> index in filter }.toMutableList()
    for (it in 0 until min(source.size, neurons.size)) {
        neurons[it] = source[it]
    }
}

fun main(args: Array<String>) {
    changeStructure("nets/nw.net", "nets/nw.net", (0..5).toList(), emptyMap(), buildNetwork(2, 2, 2, 2, 40, 3))
}