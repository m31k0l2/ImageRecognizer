import java.util.*

fun testMedianNet(nw: Network, batch: Set<Image>, teachNumbers: IntArray): Double {
    val r = batch.filter { it.index in teachNumbers }.groupBy { it.index }.map {
        (n, list) ->
            val i = teachNumbers.indexOf(n)
            n to list.map { nw.activate(it, 15.0)[i] }.average() }.toMap()
//    var sum = 0.0
    val o = teachNumbers.map {
        val y = r[it]!!
        println("$it -> ${(y * 1000).toInt() / 10.0}%")
        y
    }
    val y = o.min()!!
//    val y = sum / teachNumbers.size
    println("min успех: ${(y*1000).toInt()/10.0}%")
    println("средний успех: ${(o.average()*1000).toInt()/10.0}%")
    return (y*10000).toInt() / 10000.0
}

fun main(args: Array<String>) {
    val teachNumbers = intArrayOf(7,8,9)
    while (true) {
        print("Размер батча: ")
        val size = Scanner(System.`in`).nextInt()
        if (size < 30) return
        val name = "nw.net"
        CNetwork().load("nets/$name")?.let {
            testMedianNet(it, MNIST.buildBatch(size), teachNumbers)
        }
    }
}