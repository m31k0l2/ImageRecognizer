import java.util.logging.FileHandler
import java.util.logging.SimpleFormatter

fun clean(number: Int, teachNumbers: IntArray): MutableMap<Int, MutableList<Pair<Int, Double>>> {
    val map = mutableMapOf<Int, MutableList<Pair<Int, Double>>>()
    val fh = FileHandler("log.txt")
    log.addHandler(fh)
    fh.formatter = SimpleFormatter()
    val nw = CNetwork().load("nets/nwx.net")!!
    val testBatch = MNIST.buildBatch(500).filter { it.index in teachNumbers }.toSet()
    val initResult = testMedianNet(number, nw, testBatch)
    log.info("initResult: $initResult")
    nw.layers.forEachIndexed { layerNumber, layer ->
        if (layerNumber >= 4) return@forEachIndexed
        layer.neurons.forEachIndexed { neuronNumber, _ ->
            val test = nw.clone()
            nullNeuron(test, "nets/nw.net", layerNumber, neuronNumber)
            val res = testMedianNet(number, test, testBatch)
            if ((res*1000).toInt() <= (initResult*1000).toInt()) {
                val list = map[layerNumber]
                if (list == null) {
                    map[layerNumber] = mutableListOf()
                }
                map[layerNumber]!!.add(neuronNumber to res)
            }
            log.info("$layerNumber: $neuronNumber -> $res")
        }
    }
    println(map)
    return map
}

fun nullNeuron(nw: Network, to: String, layerNumber: Int, neuronNumber: Int) {
    val neuron = nw.layers[layerNumber].neurons[neuronNumber]
    neuron.weights = neuron.weights.map { 0.0 }.toMutableList()
    nw.save(to)
}
