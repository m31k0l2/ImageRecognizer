fun main(args: Array<String>) {
    val nw = NetworkIO().load("nets/nw589.net")!!
    val batch = MNIST.buildBatch(50)
    (0..9).forEach { n ->
        val r = batch.filter { it.index == n }.map { nw.activate(it) }.mapNotNull {
            val max = it.max()!!
            if (max > 0.8) it.indexOf(max)
            else null
        }
        println("$n -> $r")
    }
}