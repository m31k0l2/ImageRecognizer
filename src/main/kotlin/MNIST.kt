import java.io.File
import java.util.*

object MNIST {
    private val mnistTrainPath = "G:\\ih8unem\\mnist_png\\training"
    private val mnistTestPath = "G:\\ih8unem\\mnist_png\\testing"
    private val dir = File(mnistTrainPath)
    val batch: MutableList<Image> = mutableListOf()

    fun createBatch(size: Int) {
        val isEmpty = batch.isEmpty()
        for (i in 0 until size/10) {
            addToBatch()
        }
        if (isEmpty) return
        val newBatch = (0 until size/10).flatMap {
            batch.shuffle()
            (0..9).map { i -> batch.find { it.index == i }!! }
        }
        batch.clear()
        batch.addAll(newBatch.sortedBy { it.index })
    }

    fun addToBatch() {
        batch.addAll(dir.listFiles().map {
            it.listFiles().toList().shuffled().first()
        }.map { Image(it) })
        batch.sortBy { it.index }
    }

    fun next(): Image {
        val files = File(mnistTestPath).listFiles().flatMap { it.listFiles().toList() }
        return Image(files[Random().nextInt(files.size)])
    }
}