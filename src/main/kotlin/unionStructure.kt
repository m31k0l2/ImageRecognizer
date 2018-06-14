val nets = (1..2).map { "nets/789/nw$it.net" }.mapNotNull { CNetwork().load(it) }
fun main(args: Array<String>) {
    val nw = buildDoubleNetwork(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3)
    (0..5).forEach { nw.layers[it].neurons.addAll(nets[0].layers[it].neurons) }
    (0..5).forEach { nw.layers[6+it].neurons.addAll(nets[1].layers[it].neurons) }
    nw.activate(MNIST.buildBatch(10).first(), 1.0)
    nw.save("nets/nw.net")
}