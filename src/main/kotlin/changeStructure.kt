import kotlin.math.min
/*
INFO: 2: 0 -> 0.667824988107406
INFO: 3: 0 -> 0.759723628773405
INFO: 3: 1 -> 0.7315132392304711

 */
fun main(args: Array<String>) {
    changeStructure("nets/nw.net", "nets/nw.net", listOf(0,1,2,3), mapOf(
//        1 to listOf(0,2)
    ), CNetwork(6,6,4,4,60,10))
}
//0-0,1;1-0,1,2,3;2-0,2;3-2,3,5

fun changeStructure(from: String, to: String, layers: List<Int>, filters: Map<Int, List<Int>>, net: Network) {
    val nw = NetworkIO().load(from)!!
    layers.forEach {
        val filter = if (it in filters.keys) filters[it]!! else emptyList()
        changeNeurons(it, net, nw, filter)
    }
    net.activate(MNIST.buildBatch(10).first())
    NetworkIO().save(net, to)
}

private fun changeNeurons(savedLayerNumber: Int, net: Network, nw: Network, filter: List<Int>) {
    val neurons = net.layers[savedLayerNumber].neurons
    var source = nw.layers[savedLayerNumber].neurons
    if (!filter.isEmpty()) source = source.filterIndexed { index, _ -> index in filter }.toMutableList()
    for (it in 0 until min(source.size, neurons.size)) {
        neurons[it] = source[it]
    }
}