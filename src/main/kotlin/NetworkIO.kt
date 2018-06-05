import java.io.File
import java.io.FileReader
import java.io.FileWriter

class NetworkIO {
    fun save(nw: Network, name: String) {
        val file = FileWriter(File(name))
        nw.layers.forEach {
            if (it is CNNLayer) {
                file.write("CNNLayer\n")
            } else {
                file.write("layer\n")
            }
            it.neurons.forEach {
                file.write("neuron\n")
                it.weights.forEach { file.write("${(it*10000).toInt()/10000.0}\n") }
            }
        }
        file.write("end")
        file.close()
    }

    fun load(name: String): Network? {
        val f = File(name)
        if (!f.exists()) return null
        val file = FileReader(f)
        val lines = file.readLines()
        file.close()
        val nw = CNetwork()
        var layer: Layer? = null
        var neuron: Neuron? = null
        lines.forEach { line ->
            when (line) {
                "CNNLayer" -> {
                    layer?.let {
                        it.neurons.add(neuron!!)
                        nw.layers.add(it)
                        neuron = null
                    }
                    layer = CNNLayer(CNetwork.cnnDividers[nw.layers.size], CNetwork.poolerDividers[nw.layers.size]?.let { Pooler(it) })
                }
                "layer" -> {
                    layer?.let {
                        it.neurons.add(neuron!!)
                        nw.layers.add(it)
                        neuron = null
                    }
                    layer = FullConnectedLayer()
                }
                "neuron" -> {
                    neuron?.let { layer!!.neurons.add(it) }
                    neuron = Neuron()
                }
                "end" -> {
                    layer!!.neurons.add(neuron!!)
                    nw.layers.add(layer!!)
                }
                else -> neuron!!.weights.add(line.toDouble())
            }
        }
        return nw
    }
}