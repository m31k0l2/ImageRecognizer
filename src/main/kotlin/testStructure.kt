import java.util.*

fun testMedianNet(nw: Network, batch: List<Image>, teachNumbers: IntArray): Double {
    val b = teachNumbers.toList().mapNotNull { i -> batch.filter { it.index == i }.map { nw.activate(it, 15.0)[i] }.let {
        if (it.isNotEmpty()) it.average() else null
    } }
    b.forEachIndexed { index, y ->
        println("${teachNumbers[index]} -> ${(y * 1000).toInt() / 10.0}%")
    }
    val y = b.average()
    println("средний успех: ${(y*1000).toInt()/10.0}%")
    return (y*10000).toInt() / 10000.0
}

fun main(args: Array<String>) {
    while (true) {
        print("Размер батча: ")
        val size = Scanner(System.`in`).nextInt()
        if (size < 30) return
        val name = "nw.net".takeIf { size > 0 } ?: "_nw.net"
        CNetwork().load("nets/$name")?.let {
            testMedianNet(it, MNIST.buildBatch(size), IntArray(10, {it}))
        }
    }
}