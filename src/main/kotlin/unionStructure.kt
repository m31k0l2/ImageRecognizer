val nets = (1..11).map { "nets/456/nw$it.net" }.mapNotNull { CNetwork().load(it) }
fun main(args: Array<String>) {
    val nw = buildNetwork(0, 0, 0, 0, 0, 0)
    val structure = intArrayOf(2,2,2,2,40,3)
    for (i in 0..3) {
        val neurons = nets.map { it.layers[i] }.flatMap { it.neurons }.shuffled().take(structure[i])
        nw.layers[i].neurons.addAll(neurons)
    }
    for (i in 4..5) {
        val neurons = nets.map { it.layers[i] }.map { it.neurons }.shuffled().first()
        nw.layers[i].neurons.addAll(neurons)
    }
    nw.activate(MNIST.buildBatch(10).first(), 1.0)
    nw.save("nets/nw.net")
}