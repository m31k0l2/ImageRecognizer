import kotlin.math.min
/*
INFO: 0: 1 -> 0.3619117315100169
INFO: 0: 2 -> 0.9379234557125509
INFO: 1: 0 -> 0.6648372459771883
INFO: 1: 1 -> 0.6824134179675411
INFO: 1: 3 -> 0.665910143603202
INFO: 2: 3 -> 0.3342745406933549
INFO: 3: 0 -> NaN
 */
fun main(args: Array<String>) {
    changeStructure("nets/nw345.net", "nets/nw.net", listOf(0,1,2,3), mapOf(
        0 to listOf(1,2),
        1 to listOf(0,1,3),
        2 to listOf(3),
        3 to listOf(0)
    ))
}
//0-0,1;1-0,1,2,3;2-0,2;3-2,3,5

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
