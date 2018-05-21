import kotlin.math.min

fun main(args: Array<String>) {
    changeStructure("nets/nw4-9.net", "nets/nw.net", listOf(0,1,2,3), mapOf())
}

fun changeStructure(from: String, to: String, layers: List<Int>, filters: Map<Int, List<Int>>) {
    val nw = NetworkIO().load(from)!!
    val net = ImageNetEvolution().createNet()
    layers.forEach {
        val filter = if (it in filters.keys) filters[it]!! else emptyList()
        changeNeurons(it, net, nw, filter)
    }
    net.activate(MNIST.buildBatch(10).first())
    NetworkIO().save(net, to)
}

private fun changeNeurons(savedLayerNumber: Int, net: CNetwork, nw: Network, filter: List<Int>) {
    val neurons = net.layers[savedLayerNumber].neurons
    var source = nw.layers[savedLayerNumber].neurons
    if (!filter.isEmpty()) source = source.filterIndexed { index, _ -> index in filter }.toMutableList()
    for (it in 0 until min(source.size, neurons.size)) {
        neurons[it] = source[it]
    }
}
