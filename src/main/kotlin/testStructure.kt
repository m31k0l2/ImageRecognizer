import java.util.*

fun testMedianNet(number: Int, nw: Network, batch: Set<Image>, alpha: Double=15.0): Double {
    return batch.groupBy { it.index }.map { (i, list) ->
        val r = list.map { nw.activate(it, alpha)[0] }.average()
        if (i == number) r else 1 - r
    }.reduce { acc, d -> acc*d }
}

fun main(args: Array<String>) {
    while (true) {
        print("Размер батча: ")
        val size = Scanner(System.`in`).nextInt()
        if (size < 30) return
        val name = "nw.net"
        val nw = CNetwork().load("nets/$name")
        for (number in 0..9) {
            nw?.let {
                val r = testMedianNet(number, it, MNIST.buildBatch(size))
                println("$number -> $r")
            }
        }
    }
}