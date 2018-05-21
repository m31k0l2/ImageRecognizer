import java.util.logging.SimpleFormatter

fun main(args: Array<String>) {
    log.addHandler(fh)
    fh.formatter = SimpleFormatter()
    val net = ImageNetEvolution().createNet()
    val nw = NetworkIO().load("nets/nw4-9.net")!!
    val testBatch = MNIST.buildBatch(500).filter { it.index in (4..9) }
    net.layers.forEachIndexed { layerNumber, layer ->
        if (layerNumber == 4) return@forEachIndexed
        layer.neurons.forEachIndexed { neuronNumber, _ ->
            val test = nw.clone()
            nullNeuron(test, "nets/nw.net", layerNumber, neuronNumber)
            val res = testMedianNet(test, testBatch)
            log.info("$layerNumber: $neuronNumber -> $res")
        }
    }
}

fun nullNeuron(nw: Network, to: String, layerNumber: Int, neuronNumber: Int) {
    val neuron = nw.layers[layerNumber].neurons[neuronNumber]
    neuron.weights = neuron.weights.map { 0.0 }.toMutableList()
    NetworkIO().save(nw, to)
}
