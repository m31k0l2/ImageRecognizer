import java.awt.Toolkit
import java.util.logging.*

val log: Logger = Logger.getLogger("logger")

fun beep() {
    for (i in 1..60) {
        Toolkit.getDefaultToolkit().beep()
        Thread.sleep(1000)
    }
}

fun getStructure(path: String): IntArray {
    val nw = NetworkIO().load(path) ?: return emptyArray<Int>().toIntArray()
    return nw.layers.map { it.neurons.size }.toIntArray()
}

fun rebuild(teachNumbers: IntArray, hiddenLayerNeurons: Int=40) {
    val map = clean(teachNumbers)
    var list = mutableListOf<Int>()
    for (i in 0..3) {
        map[i]?.let { list.add(it.size) } ?: list.add(1)
    }
    if (list.reduce { acc, i -> acc * i } == 1) {
        list = mutableListOf(1, 1, 2, 2)
    }
    list.add(hiddenLayerNeurons)
    list.add(10)
    println(list)
    map.map { it.key to it.value.map { it.first } }.toMap()
    changeStructure("nets/nwx.net", "nets/nw.net", listOf(0,1,2,3), map.map { it.key to it.value.map { it.first } }.toMap(), buildNetwork(list.toIntArray(), 1.0))
}

fun saveAs(from: String, to: String) {
    val nw = NetworkIO().load(from)!!
    NetworkIO().save(nw, to)
}

fun setupLog(log: Logger) {
    val fh = FileHandler("log.txt")
    log.addHandler(fh)
    class NoTimeStampFormatter : SimpleFormatter() {
        override fun format(record: LogRecord?): String {
            return if (record?.level == Level.INFO) {
                record?.message + "\r\n"
            } else {
                super.format(record)
            }
        }
    }
    fh.formatter = NoTimeStampFormatter()
}