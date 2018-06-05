import java.util.*

fun testNet(nw: Network, batch: List<Image>): Double {
    var y = 0.0
    var counter = 0
    batch.sortedBy { it.index }.forEach {
        val o = nw.activate(it)
        if (it.index >= o.size) return@forEach
        val result = o[it.index]
        if (result != o.max()) {
            println("${it.index} -> ${(result * 10000).toInt() / 100.0}% [${o.indexOf(o.max())} (${(o.max()!! * 10000).toInt() / 100.0}%)]")
        } else {
            counter++
            println("${it.index} -> ${(result * 10000).toInt() / 100.0}% [OK]")
        }
        y += result
    }
    y = (y*1000).toInt()/(batch.size*10.0)
    println("средний успех: $y%")
    println("$counter / ${batch.size}")
    return y
}

fun testMedianNet(nw: Network, batch: List<Image>, teachNumbers: IntArray): Double {
    val b = teachNumbers.toList().mapNotNull { i -> batch.filter { it.index == i }.map { nw.activate(it)[i] }.let {
        if (it.isNotEmpty()) it.sorted()[it.size / 2] else null
    } }
    b.forEachIndexed { index, y ->
        println("${teachNumbers[index]} -> ${(y * 1000).toInt() / 10.0}%")
    }
    val y = b.average()
    println("средний успех: ${(y*1000).toInt()/10.0}%")
    return (y*10000).toInt() / 10000.0
}

fun main(args: Array<String>) {
    Network.useSigma = true
    Neuron.alpha = 15.0
    while (true) {
        print("Размер батча: ")
        val size = Scanner(System.`in`).nextInt()
        if (size < 30) return
        val name = "nw.net".takeIf { size > 0 } ?: "_nw.net"
        NetworkIO().load("nets/$name")?.let { testMedianNet(it, MNIST.buildBatch(size), IntArray(10, {it})) }
    }
}