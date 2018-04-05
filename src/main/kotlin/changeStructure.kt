import java.io.File

val image = MNIST.buildBatch(10).first()

fun main(args: Array<String>) {
    val dir = File("nets")
    dir.listFiles().forEach {
        changeStructure(it, "nets/nw.net")
    }
}

fun changeStructure(file: File, name: String) {
    val nw = NetworkIO().load(file.absolutePath)!!
    val net = CNetwork(0, 0, 0, 0, 0, 10)
    net.layers[0].neurons.addAll(nw.layers[0].neurons)
    net.layers[1].neurons.addAll(nw.layers[1].neurons)
    net.layers[2].neurons.addAll(nw.layers[2].neurons)
    net.layers[3].neurons.addAll(nw.layers[3].neurons)
    net.layers[4].neurons.addAll(nw.layers[4].neurons)
    net.activate(image)
    NetworkIO().save(net, name)
}
