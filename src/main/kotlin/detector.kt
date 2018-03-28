fun main(args: Array<String>) {
    val detector = FNetwork(10)
    val batch = MNIST.buildBatch(10)

    batch.forEach {
        val r = detector.activate(it)
        println(r)
    }
}